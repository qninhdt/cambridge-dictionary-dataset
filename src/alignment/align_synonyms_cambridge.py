import sqlite3
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import argparse
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from src.utils.nlp import get_match_keys

# Custom Bi-Encoder wrapper
class BiEncoder:
    def __init__(self, model_name, device='cpu'):
        print(f"Loading Bi-Encoder {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(device)
        self.device = device
        print("Bi-Encoder loaded successfully.")

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, texts, batch_size=32):
        if not texts:
            return np.empty((0, 1024))
        all_embeddings = []
        # Wrap range with tqdm to display progress bar
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating Embeddings"):
            batch_texts = texts[i:i+batch_size]
            try:
                with torch.no_grad():
                    inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = self.model(**inputs)
                    embeddings = self.mean_pooling(outputs, inputs['attention_mask'])
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    all_embeddings.append(embeddings.cpu().numpy())
            except torch.OutOfMemoryError:
                print("\n[Warning] CUDA OOM encountered. Falling back to CPU for this batch.")
                torch.cuda.empty_cache()
                # Run this batch on CPU
                original_device = self.device
                self.model.to('cpu')
                with torch.no_grad():
                    inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
                    outputs = self.model(**inputs)
                    embeddings = self.mean_pooling(outputs, inputs['attention_mask'])
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    all_embeddings.append(embeddings.numpy())
                self.model.to(original_device)
        return np.vstack(all_embeddings)

# Helper function to map POS to standardized character for matching
def normalize_pos(pos):
    if not pos:
        return ""
    pos = pos.lower().strip()
    if 'noun' in pos or pos == 'n': return 'n'
    if 'verb' in pos or pos == 'v': return 'v'
    if 'adjective' in pos or pos == 'adj' or pos == 'a': return 'a'
    if 'adverb' in pos or pos == 'adv' or pos == 'r': return 'r'
    return pos


def main():
    parser = argparse.ArgumentParser(description="Map Cambridge synonyms to target sense IDs directly in cambridge.db.")
    parser.add_argument("--db_path", type=str, default="data/cambridge.db", help="Path to cambridge.db file.")
    parser.add_argument("--alignment_db_path", type=str, default="data/dictionary_alignment.db", help="Path to dictionary_alignment.db.")
    parser.add_argument("--model_name", type=str, default="mixedbread-ai/mxbai-embed-large-v1", help="Hugging Face embedding model name.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for generating embeddings.")
    parser.add_argument("--device", type=str, default=None, help="Device to run PyTorch model ('cuda', 'cpu', 'mps').")
    parser.add_argument("--dry_run", action="store_true", help="Dry run mode. Perform all computations but do not update the DB.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild all mappings (including already matched synonyms).")
    args = parser.parse_args()

    if not os.path.exists(args.db_path):
        print(f"Error: Database file not found at {args.db_path}")
        sys.exit(1)

    print(f"Connecting to database: {args.db_path}")
    conn = sqlite3.connect(args.db_path)
    cursor = conn.cursor()

    # Step 1: Count antonyms
    cursor.execute("SELECT COUNT(*) FROM sense_synonyms WHERE is_antonym = 1")
    antonym_count = cursor.fetchone()[0]
    print(f"Found {antonym_count} antonym relations in DB (will be skipped).")

    # Step 2: Add column target_sense_id if not exists
    cursor.execute("PRAGMA table_info(sense_synonyms)")
    columns = [col[1] for col in cursor.fetchall()]
    if "target_sense_id" not in columns:
        if args.dry_run:
            print("[Dry-run] Would add column 'target_sense_id' to 'sense_synonyms' table.")
        else:
            print("Adding column 'target_sense_id' to 'sense_synonyms' table...")
            cursor.execute("ALTER TABLE sense_synonyms ADD COLUMN target_sense_id INTEGER REFERENCES senses(id)")
            conn.commit()

    # Reset existing mappings if rebuilding
    if args.rebuild and not args.dry_run:
        print("Rebuild requested: resetting all existing target_sense_id mappings to NULL...")
        cursor.execute("UPDATE sense_synonyms SET target_sense_id = NULL")
        conn.commit()

    # Step 3: Load all senses into memory index for quick candidate retrieval
    print("Indexing all Cambridge senses...")
    cursor.execute("""
        SELECT s.id, s.definition, w.display_form, w.word, e.pos
        FROM senses s
        JOIN entries e ON s.entry_id = e.id
        JOIN words w ON e.word_id = w.id
    """)
    senses_by_word_pos = defaultdict(list)
    senses_defn_cache = {}
    for s_id, defn, display, slug, pos in cursor.fetchall():
        pos_norm = normalize_pos(pos)
        senses_defn_cache[s_id] = defn
        
        # Populate candidates using get_match_keys for robust string matching
        keys = set()
        if display:
            keys.update(get_match_keys(display))
        if slug:
            keys.update(get_match_keys(slug))
            
        for k in keys:
            senses_by_word_pos[(k, pos_norm)].append(s_id)

    # Build inverted index for close phrase matching
    print("Building inverted index for close matches...")
    stop_words = {
        'and', 'the', 'a', 'an', 'for', 'with', 'to', 'in', 'on', 'at', 
        'by', 'of', 'about', 'out', 'up', 'down', 'off', 'over', 'under', 
        'be', 'been', 'is', 'are', 'was', 'were', 'have', 'has', 'had', 'do', 
        'does', 'did', 'some', 'any', 'that', 'this', 'these', 'those'
    }
    inverted_index = defaultdict(list)
    for idx_key, p_key in senses_by_word_pos.keys():
        idx_tokens = set(idx_key.split())
        important_idx_tokens = {t for t in idx_tokens if len(t) > 2 and t not in stop_words}
        for t in important_idx_tokens:
            inverted_index[t].append((idx_key, p_key))

    # Step 4: Pre-fetch synonyms and examples for rule check and example concatenation
    cursor.execute("SELECT sense_id, synonym FROM sense_synonyms WHERE is_antonym = 0")
    synonyms_by_sense = defaultdict(set)
    for s_id, synonym in cursor.fetchall():
        synonyms_by_sense[s_id].add(synonym.lower().strip())

    cursor.execute("SELECT sense_id, example FROM examples")
    examples_by_sense = defaultdict(list)
    for s_id, example in cursor.fetchall():
        examples_by_sense[s_id].append(example.strip())

    # Step 5: Optional WordNet Enrichment load
    unified_id_by_cam_sense = {}
    wn_links_by_unified_id = defaultdict(list)
    wordnet_defn_cache = {}
    use_enrichment = False

    if os.path.exists(args.alignment_db_path):
        print(f"Alignment DB found at {args.alignment_db_path}. Loading WordNet enrichment data...")
        try:
            conn_align = sqlite3.connect(args.alignment_db_path)
            c_align = conn_align.cursor()
            
            c_align.execute("SELECT id, cambridge_sense_id FROM unified_senses WHERE cambridge_sense_id IS NOT NULL")
            for u_id, cam_sid in c_align.fetchall():
                unified_id_by_cam_sense[cam_sid] = u_id
                
            c_align.execute("SELECT unified_sense_id, wordnet_id FROM unified_sense_wordnet_links")
            for us_id, wn_id in c_align.fetchall():
                wn_links_by_unified_id[us_id].append(wn_id)
                
            conn_align.close()
            
            import wn
            oewn = wn.Wordnet('oewn:2024')
            use_enrichment = True
            print("WordNet enrichment successfully enabled.")
        except Exception as e:
            print(f"[Warning] Failed to load WordNet enrichment: {e}. Falling back to raw Cambridge definitions.")
    else:
        print("[Info] Alignment DB not found. Continuing with raw Cambridge definitions.")

    def get_wordnet_definition(wn_id):
        if wn_id in wordnet_defn_cache:
            return wordnet_defn_cache[wn_id]
        try:
            ss = oewn.synset(wn_id)
            defn = ss.definition()
            wordnet_defn_cache[wn_id] = defn
            return defn
        except Exception:
            wordnet_defn_cache[wn_id] = ""
            return ""

    def get_enriched_info(cam_sense_id):
        cam_syns = set(synonyms_by_sense.get(cam_sense_id, set()))
        wn_defns = []
        if use_enrichment:
            u_id = unified_id_by_cam_sense.get(cam_sense_id)
            if u_id:
                wn_ids = wn_links_by_unified_id.get(u_id, [])
                for wn_id in wn_ids:
                    # Also fetch WordNet synonyms to enrich rule matching
                    try:
                        ss = oewn.synset(wn_id)
                        cam_syns.update(w.lemma().lower().strip() for w in ss.words())
                    except Exception:
                        pass
                    wn_defn = get_wordnet_definition(wn_id)
                    if wn_defn:
                        wn_defns.append(wn_defn)
        return cam_syns, wn_defns

    # Step 6: Fetch synonyms relations to map
    print("Loading synonyms relationships from cambridge.db...")
    query = """
        SELECT ss.id, ss.sense_id, ss.synonym, w.display_form, e.pos
        FROM sense_synonyms ss
        JOIN senses s ON ss.sense_id = s.id
        JOIN entries e ON s.entry_id = e.id
        JOIN words w ON e.word_id = w.id
        WHERE ss.is_antonym = 0
    """
    if not args.rebuild:
        query += " AND ss.target_sense_id IS NULL"
    cursor.execute(query)
    relations = cursor.fetchall()

    senses_needing_embeddings = set()
    mappings_to_resolve_ai = []
    final_mappings_to_update = []

    print("\n--- PHASE 1: COLLECTING & FILTERING RULES ---")
    skipped_no_candidates = 0
    rule_resolved = 0

    for rel_id, src_sense_id, target_text, src_word, src_pos_raw in tqdm(relations, desc="Checking Rules"):
        pos_norm = normalize_pos(src_pos_raw)
        if not pos_norm:
            continue
            
        # Generate match keys for target synonym
        target_keys = get_match_keys(target_text)
        
        # Support wildcard POS matching for collocations, idioms, phrases, expressions, and closed classes
        wildcards = {
            'collocation', 'phrase', 'idiom', 'expression', 'exclamation',
            'preposition', 'conjunction', 'determiner', 'pronoun', 'number',
            'prefix', 'suffix'
        }
        search_pos = {pos_norm}
        if pos_norm in wildcards:
            search_pos.update({'n', 'v', 'a', 'r'})
            search_pos.update(wildcards)
        else:
            search_pos.update(wildcards)

        tgt_senses = []
        for tk in target_keys:
            # 1. Exact match lookup
            for p_key in search_pos:
                tgt_senses.extend(senses_by_word_pos.get((tk, p_key), []))
                
            # 2. Close match lookup (Jaccard >= 0.60 and Containment >= 0.99) using inverted index
            tk_tokens = set(tk.split())
            important_tokens = {t for t in tk_tokens if len(t) > 2 and t not in stop_words}
            
            eval_keys = set()
            for t in important_tokens:
                if t in inverted_index:
                    eval_keys.update(inverted_index[t])
                    
            for idx_key, p_key in eval_keys:
                if p_key not in search_pos:
                    continue
                
                idx_tokens = set(idx_key.split())
                intersection = len(tk_tokens.intersection(idx_tokens))
                union = len(tk_tokens.union(idx_tokens))
                jaccard = intersection / union
                containment = intersection / min(len(tk_tokens), len(idx_tokens))
                
                if jaccard >= 0.60 and containment >= 0.99:
                    tgt_senses.extend(senses_by_word_pos[(idx_key, p_key)])
                    
        tgt_senses = list(set(tgt_senses))

        if not tgt_senses:
            skipped_no_candidates += 1
            continue

        # Rule 1: Single Candidate -> Direct Link
        if len(tgt_senses) == 1:
            final_mappings_to_update.append((tgt_senses[0], rel_id))
            rule_resolved += 1
            continue

        src_syns, src_wn_defns = get_enriched_info(src_sense_id)

        # Build candidate features
        candidates_data = []
        for s_id in tgt_senses:
            tgt_syns, tgt_wn_defns = get_enriched_info(s_id)
            
            is_mutual = False
            src_keys = get_match_keys(src_word)
            for src_key in src_keys:
                normalized_tgt_syns = set()
                for ts in tgt_syns:
                    normalized_tgt_syns.update(get_match_keys(ts))
                if src_key in normalized_tgt_syns:
                    is_mutual = True
                    break
                    
            overlap_score = len(src_syns.intersection(tgt_syns))
            candidates_data.append({
                'sense_id': s_id,
                'is_mutual': is_mutual,
                'overlap_score': overlap_score
            })

        # Apply Hierarchical Rule Selection
        mutual_candidates = [c for c in candidates_data if c['is_mutual']]
        if mutual_candidates:
            max_mutual_overlap = max(c['overlap_score'] for c in mutual_candidates)
            best_candidates = [c for c in mutual_candidates if c['overlap_score'] == max_mutual_overlap]
        else:
            max_overall_overlap = max(c['overlap_score'] for c in candidates_data)
            if max_overall_overlap > 0:
                best_candidates = [c for c in candidates_data if c['overlap_score'] == max_overall_overlap]
            else:
                best_candidates = candidates_data
                max_overall_overlap = 0

        need_dl = len(best_candidates) > 1 or (not mutual_candidates and max_overall_overlap == 0)

        if need_dl:
            senses_needing_embeddings.add(src_sense_id)
            for c in candidates_data:
                senses_needing_embeddings.add(c['sense_id'])
                
            mappings_to_resolve_ai.append({
                'rel_id': rel_id,
                'src_sense_id': src_sense_id,
                'best_candidates': best_candidates
            })
        else:
            final_mappings_to_update.append((best_candidates[0]['sense_id'], rel_id))
            rule_resolved += 1

    print(f"\nPHASE 1 SUMMARY:")
    print(f"  - Total synonyms relations:       {len(relations)}")
    print(f"  - Skipped (no target candidates): {skipped_no_candidates}")
    print(f"  - Rule-resolved:                  {rule_resolved}")
    print(f"  - Fallback to AI needed:          {len(mappings_to_resolve_ai)}")
    print(f"  - Unique senses requiring AI emb: {len(senses_needing_embeddings)}")

    if not mappings_to_resolve_ai:
        print("\nAll relations resolved by rules. Skipping AI phase.")
    else:
        print("\n--- PHASE 2: GENERATING EMBEDDINGS (BATCHED WITH EXAMPLES) ---")
        device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
        encoder = BiEncoder(args.model_name, device=device)

        # Build enriched context string for required senses
        ordered_ids = list(senses_needing_embeddings)
        context_texts = []
        for s_id in ordered_ids:
            base_defn = senses_defn_cache[s_id]
            _, wn_defns = get_enriched_info(s_id)
            
            enriched = base_defn
            if wn_defns:
                enriched += " [WordNet: " + "; ".join(wn_defns) + "]"
                
            # Add Cambridge examples
            if s_id in examples_by_sense:
                enriched += " Examples: " + "; ".join(examples_by_sense[s_id])
                
            context_texts.append(enriched)

        print(f"Encoding {len(context_texts)} enriched contexts...")
        embeddings_matrix = encoder.encode(context_texts, batch_size=args.batch_size)
        
        # Build embedding lookup dictionary
        embeddings_by_id = {}
        for idx, s_id in enumerate(ordered_ids):
            embeddings_by_id[s_id] = embeddings_matrix[idx]
        print("Embeddings cached in memory.")

        print("\n--- PHASE 3: RESOLVING TIES WITH COSINE SIMILARITY ---")
        ai_resolved = 0
        for task in tqdm(mappings_to_resolve_ai, desc="Running AI matching"):
            rel_id = task['rel_id']
            src_sense_id = task['src_sense_id']
            best_candidates = task['best_candidates']
            
            src_emb = embeddings_by_id[src_sense_id]
            
            best_id = None
            max_sim = -1.0
            
            for c in best_candidates:
                c_id = c['sense_id']
                c_emb = embeddings_by_id[c_id]
                sim = float(np.dot(src_emb, c_emb))
                if sim > max_sim:
                    max_sim = sim
                    best_id = c_id
                    
            final_mappings_to_update.append((best_id, rel_id))
            ai_resolved += 1
        print(f"AI Phase resolved: {ai_resolved} tasks.")

    print("\n--- PHASE 4: WRITING RESULTS TO CAMBRIDGE DATABASE ---")
    if args.dry_run:
        print(f"[Dry-run] Would update {len(final_mappings_to_update)} rows in 'sense_synonyms' table.")
    else:
        cursor.executemany("""
            UPDATE sense_synonyms
            SET target_sense_id = ?
            WHERE id = ?
        """, final_mappings_to_update)
        conn.commit()
        print(f"Successfully updated {len(final_mappings_to_update)} synonym rows in database.")

    # Create index on target_sense_id for rapid querying
    if not args.dry_run:
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_synonyms_target_sense ON sense_synonyms(target_sense_id)")
        conn.commit()

    print("\n" + "="*80)
    print("CAMBRIDGE SYNONYMS DIRECT MAPPING COMPLETED")
    print("="*80)
    print(f"  - Database file:            {args.db_path}")
    print(f"  - Total synonyms mapped:    {len(final_mappings_to_update)}")
    print(f"  - Rule-based mappings:      {rule_resolved}")
    print(f"  - AI-based mappings:        {len(mappings_to_resolve_ai)}")
    print(f"  - Model used:               {args.model_name}")
    print("="*80 + "\n")

    conn.close()

if __name__ == "__main__":
    main()
