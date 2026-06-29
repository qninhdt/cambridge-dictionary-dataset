#!/usr/bin/env python3
import sqlite3
import os
import re
import json
from tqdm import tqdm
import wn

def slugify(text):
    text = text.lower().strip()
    # Replace non-word characters with empty space
    text = re.sub(r'[^\w\s-]', '', text)
    # Replace spaces and underscores with hyphens
    text = re.sub(r'[\s_]+', '-', text)
    # Replace multiple hyphens with single hyphen
    text = re.sub(r'-+', '-', text)
    return text.strip('-')

def map_pos_to_enum(pos_str):
    if not pos_str:
        return 'other'
    pos = pos_str.lower().strip()
    if 'noun' in pos or pos == 'n':
        return 'noun'
    if 'verb' in pos or pos == 'v':
        return 'verb'
    if 'adjective' in pos or pos == 'adj' or pos == 'a' or pos == 's':
        return 'adjective'
    if 'adverb' in pos or pos == 'adv' or pos == 'r':
        return 'adverb'
    return 'other'

def map_pos_char_to_enum(pos_char):
    if not pos_char:
        return 'other'
    pos = pos_char.lower().strip()
    if pos == 'n':
        return 'noun'
    if pos == 'v':
        return 'verb'
    if pos in ('a', 's'):
        return 'adjective'
    if pos == 'r':
        return 'adverb'
    return 'other'

def guess_entry_type(word_str, pos_enum):
    word_str = word_str.strip()
    if ' ' in word_str or '-' in word_str:
        if pos_enum == 'verb':
            return 'phrasal_verb'
        return 'phrase'
    return 'word'

def main():
    cambridge_db_path = "data/cambridge.db"
    alignment_db_path = "data/dictionary_alignment.db"
    final_db_path = "data/dictionary.db"

    # Remove existing final DB if any to ensure clean build
    if os.path.exists(final_db_path):
        print(f"Removing existing database {final_db_path}...")
        os.remove(final_db_path)

    print("Connecting to source databases...")
    conn_cam = sqlite3.connect(cambridge_db_path)
    cur_cam = conn_cam.cursor()
    conn_align = sqlite3.connect(alignment_db_path)
    cur_align = conn_align.cursor()

    print("Connecting to target database...")
    conn_final = sqlite3.connect(final_db_path)
    cur_final = conn_final.cursor()

    # Enable foreign keys
    cur_final.execute("PRAGMA foreign_keys = ON;")

    print("\n--- PHASE 0: INITIALIZING DATABASE SCHEMA ---")
    cur_final.executescript("""
    CREATE TABLE words (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      slug TEXT NOT NULL UNIQUE,
      word TEXT NOT NULL,
      entry_type TEXT DEFAULT 'word' CHECK (entry_type IN ('word', 'phrasal_verb', 'idiom', 'phrase', 'expression')),
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE entries (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      word_id INTEGER NOT NULL REFERENCES words(id) ON DELETE CASCADE,
      pos TEXT NOT NULL CHECK (pos IN ('noun', 'verb', 'adjective', 'adverb', 'other')),
      pronunciation_uk TEXT,
      pronunciation_us TEXT,
      audio_uk_url TEXT,
      audio_us_url TEXT
    );

    CREATE TABLE entry_inflections (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      entry_id INTEGER NOT NULL REFERENCES entries(id) ON DELETE CASCADE,
      form_type TEXT NOT NULL CHECK (form_type IN ('plural', 'past_tense', 'past_participle', 'present_participle', 'third_person_singular')),
      inflected_form TEXT NOT NULL
    );

    CREATE TABLE senses (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      entry_id INTEGER NOT NULL REFERENCES entries(id) ON DELETE CASCADE,
      definition TEXT NOT NULL,
      cefr_level TEXT CHECK (cefr_level IN ('A1', 'A2', 'B1', 'B2', 'C1', 'C2')),
      guideword TEXT,
      grammar TEXT,
      domain TEXT,
      labels TEXT DEFAULT '[]'
    );

    CREATE TABLE alternative_definitions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      sense_id INTEGER NOT NULL REFERENCES senses(id) ON DELETE CASCADE,
      definition TEXT NOT NULL,
      source TEXT NOT NULL CHECK (source IN ('cambridge', 'wordnet')),
      wordnet_id TEXT
    );

    CREATE TABLE synonyms_antonyms (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      sense_id INTEGER NOT NULL REFERENCES senses(id) ON DELETE CASCADE,
      target_word TEXT NOT NULL,
      target_sense_id INTEGER REFERENCES senses(id) ON DELETE SET NULL,
      is_antonym BOOLEAN DEFAULT 0 CHECK (is_antonym IN (0, 1)),
      source TEXT NOT NULL CHECK (source IN ('cambridge', 'wordnet'))
    );

    CREATE TABLE examples (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      sense_id INTEGER NOT NULL REFERENCES senses(id) ON DELETE CASCADE,
      example_text TEXT NOT NULL,
      collocation TEXT,
      source TEXT NOT NULL CHECK (source IN ('cambridge', 'wordnet'))
    );

    CREATE TABLE collocations (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      word_id INTEGER NOT NULL REFERENCES words(id) ON DELETE CASCADE,
      collocation TEXT NOT NULL,
      example TEXT
    );

    CREATE TABLE semantic_relations (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      source_sense_id INTEGER NOT NULL REFERENCES senses(id) ON DELETE CASCADE,
      target_sense_id INTEGER NOT NULL REFERENCES senses(id) ON DELETE CASCADE,
      relation_type TEXT NOT NULL CHECK (relation_type IN (
        'hypernym', 'hyponym', 'part_meronym', 'member_holonym', 'cause', 'entailment',
        'similar_to', 'attribute', 'derivationally_related', 'domain_topic', 'domain_region', 'domain_usage',
        'related_to', 'form_of', 'is_a', 'part_of', 'has_a', 'used_for', 'capable_of', 'at_location',
        'causes', 'has_property', 'has_prerequisite', 'has_subevent', 'has_first_subevent', 'has_last_subevent',
        'motivated_by_goal', 'desires', 'created_by', 'synonym', 'antonym', 'distinct_from', 'derived_from', 'defined_as'
      )),
      source TEXT NOT NULL CHECK (source IN ('wordnet', 'cambridge', 'conceptnet'))
    );

    CREATE TABLE topics (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      slug TEXT NOT NULL UNIQUE,
      title TEXT NOT NULL
    );

    CREATE TABLE sense_topics (
      sense_id INTEGER REFERENCES senses(id) ON DELETE CASCADE,
      topic_id INTEGER REFERENCES topics(id) ON DELETE CASCADE,
      PRIMARY KEY (sense_id, topic_id)
    );

    CREATE INDEX idx_words_slug ON words(slug);
    CREATE INDEX idx_words_word ON words(word);
    CREATE INDEX idx_entries_word_id ON entries(word_id);
    CREATE INDEX idx_entries_pos ON entries(pos);
    CREATE INDEX idx_entry_inflections_inflected_form ON entry_inflections(inflected_form);
    CREATE INDEX idx_entry_inflections_entry_id ON entry_inflections(entry_id);
    CREATE INDEX idx_senses_entry_id ON senses(entry_id);
    CREATE INDEX idx_senses_cefr_level ON senses(cefr_level);
    CREATE INDEX idx_synonyms_antonyms_sense_id ON synonyms_antonyms(sense_id);
    CREATE INDEX idx_synonyms_antonyms_target_word ON synonyms_antonyms(target_word);
    CREATE INDEX idx_synonyms_antonyms_target_sense_id ON synonyms_antonyms(target_sense_id);
    CREATE INDEX idx_synonyms_antonyms_is_antonym ON synonyms_antonyms(is_antonym);
    CREATE INDEX idx_synonyms_antonyms_source ON synonyms_antonyms(source);
    CREATE INDEX idx_examples_sense_id ON examples(sense_id);
    CREATE INDEX idx_examples_source ON examples(source);
    CREATE INDEX idx_collocations_word_id ON collocations(word_id);
    CREATE INDEX idx_semantic_relations_source_sense_id ON semantic_relations(source_sense_id);
    CREATE INDEX idx_semantic_relations_target_sense_id ON semantic_relations(target_sense_id);
    CREATE INDEX idx_semantic_relations_relation_type ON semantic_relations(relation_type);
    CREATE INDEX idx_semantic_relations_source ON semantic_relations(source);
    CREATE INDEX idx_topics_slug ON topics(slug);
    CREATE INDEX idx_alternative_definitions_sense_id ON alternative_definitions(sense_id);
    CREATE INDEX idx_alternative_definitions_source ON alternative_definitions(source);
    """)
    conn_final.commit()
    print("Database schema and indexes initialized.")

    # In-memory mapping structures to keep track of conversions
    # Final word registry: display_form -> final_word_id
    word_display_to_final_id = {}
    # Slug registry to ensure uniqueness
    word_slugs = set()
    # Cambridge word ID -> Final word ID
    cam_word_id_to_final_id = {}

    print("\n--- PHASE 1: MIGRATING WORDS ---")
    # 1. Load words from Cambridge
    cur_cam.execute("SELECT id, word, display_form, entry_type FROM words")
    cam_words = cur_cam.fetchall()
    
    words_to_insert = []
    for cam_id, slug, display, entry_type in cam_words:
        clean_display = display if display else slug.replace('-', ' ')
        clean_slug = slugify(slug)
        if clean_slug in word_slugs:
            # Suffix if collision
            suffix = 1
            while f"{clean_slug}-{suffix}" in word_slugs:
                suffix += 1
            clean_slug = f"{clean_slug}-{suffix}"
            
        word_slugs.add(clean_slug)
        words_to_insert.append((clean_slug, clean_display, entry_type, cam_id))

    # Bulk insert Cambridge words
    for slug, display, entry_type, cam_id in words_to_insert:
        cur_final.execute(
            "INSERT INTO words (slug, word, entry_type) VALUES (?, ?, ?)",
            (slug, display, entry_type)
        )
        final_id = cur_final.lastrowid
        word_display_to_final_id[display.lower()] = final_id
        cam_word_id_to_final_id[cam_id] = final_id

    print(f"Migrated {len(words_to_insert)} words from Cambridge.")

    # 2. Check and migrate any words from unified_senses not yet in the registry
    cur_align.execute("SELECT DISTINCT word, pos FROM unified_senses")
    all_senses_words = cur_align.fetchall()
    
    wn_added = 0
    for display, pos in all_senses_words:
        display_lower = display.lower().strip()
        if display_lower not in word_display_to_final_id:
            clean_slug = slugify(display)
            if clean_slug in word_slugs:
                # Suffix if collision
                suffix = 1
                while f"{clean_slug}-{suffix}" in word_slugs:
                    suffix += 1
                clean_slug = f"{clean_slug}-{suffix}"
            
            word_slugs.add(clean_slug)
            pos_enum = map_pos_char_to_enum(pos) if len(pos) == 1 else map_pos_to_enum(pos)
            guessed_type = guess_entry_type(display, pos_enum)
            
            cur_final.execute(
                "INSERT INTO words (slug, word, entry_type) VALUES (?, ?, ?)",
                (clean_slug, display, guessed_type)
            )
            final_id = cur_final.lastrowid
            word_display_to_final_id[display_lower] = final_id
            wn_added += 1

    conn_final.commit()
    print(f"Added {wn_added} new words from unified_senses to the registry.")

    # Final Entry mapping: (final_word_id, pos_enum) -> final_entry_id
    final_entries_registry = {}
    # Cambridge entry ID -> Final entry ID
    cam_entry_id_to_final_id = {}

    print("\n--- PHASE 2: MIGRATING ENTRIES ---")
    # Get all entries from Cambridge to merge
    cur_cam.execute("SELECT id, word_id, pos, pronunciation_uk, pronunciation_us, audio_uk_url, audio_us_url FROM entries")
    cam_entries = cur_cam.fetchall()

    # Pre-group phonetics/audio by (final_word_id, mapped_pos)
    entry_groups = {} # (final_word_id, pos_enum) -> list of entries info
    for cam_ent_id, cam_w_id, pos, pron_uk, pron_us, audio_uk, audio_us in cam_entries:
        final_w_id = cam_word_id_to_final_id.get(cam_w_id)
        if not final_w_id:
            continue
        pos_enum = map_pos_to_enum(pos)
        key = (final_w_id, pos_enum)
        if key not in entry_groups:
            entry_groups[key] = []
        entry_groups[key].append({
            'cam_id': cam_ent_id,
            'pron_uk': pron_uk,
            'pron_us': pron_us,
            'audio_uk': audio_uk,
            'audio_us': audio_us
        })

    # Create entries in target database
    for (final_w_id, pos_enum), group in entry_groups.items():
        # Pick the first non-null values
        pron_uk = next((e['pron_uk'] for e in group if e['pron_uk']), None)
        pron_us = next((e['pron_us'] for e in group if e['pron_us']), None)
        audio_uk = next((e['audio_uk'] for e in group if e['audio_uk']), None)
        audio_us = next((e['audio_us'] for e in group if e['audio_us']), None)

        cur_final.execute(
            "INSERT INTO entries (word_id, pos, pronunciation_uk, pronunciation_us, audio_uk_url, audio_us_url) VALUES (?, ?, ?, ?, ?, ?)",
            (final_w_id, pos_enum, pron_uk, pron_us, audio_uk, audio_us)
        )
        final_ent_id = cur_final.lastrowid
        final_entries_registry[(final_w_id, pos_enum)] = final_ent_id

        # Map all cambridge entry IDs in this group to the single final entry ID
        for e in group:
            cam_entry_id_to_final_id[e['cam_id']] = final_ent_id

    # Create entries for all word/pos combinations from unified_senses if they don't have them
    for display, pos in all_senses_words:
        final_w_id = word_display_to_final_id[display.lower().strip()]
        pos_enum = map_pos_char_to_enum(pos) if len(pos) == 1 else map_pos_to_enum(pos)
        key = (final_w_id, pos_enum)
        if key not in final_entries_registry:
            cur_final.execute(
                "INSERT INTO entries (word_id, pos, pronunciation_uk, pronunciation_us, audio_uk_url, audio_us_url) VALUES (?, ?, NULL, NULL, NULL, NULL)",
                (final_w_id, pos_enum)
            )
            final_ent_id = cur_final.lastrowid
            final_entries_registry[key] = final_ent_id

    conn_final.commit()
    print(f"Consolidated and migrated {len(final_entries_registry)} word entries.")

    print("\n--- PHASE 3: MIGRATING ENTRY INFLECTIONS ---")
    cur_cam.execute("SELECT entry_id, form_type, inflected_form FROM entry_inflections")
    cam_inflections = cur_cam.fetchall()

    inflections_added = 0
    seen_inflections = set() # (entry_id, form_type, inflected_form)
    
    for cam_ent_id, form_type, inflected_form in cam_inflections:
        final_ent_id = cam_entry_id_to_final_id.get(cam_ent_id)
        if not final_ent_id:
            continue

        # Get final pos of this entry
        cur_final.execute("SELECT pos FROM entries WHERE id = ?", (final_ent_id,))
        pos_res = cur_final.fetchone()
        entry_pos = pos_res[0] if pos_res else 'other'

        # Map form_type to inflection_type enum
        mapped_forms = []
        ft = form_type.lower().strip() if form_type else ""
        if ft == 'plural':
            mapped_forms.append('plural')
        elif ft == 'past tense' or ft == 'past':
            mapped_forms.append('past_tense')
        elif ft == 'past participle':
            mapped_forms.append('past_participle')
        elif ft == 'present participle':
            mapped_forms.append('present_participle')
        elif ft == 'past tense and past participle':
            mapped_forms.append('past_tense')
            mapped_forms.append('past_participle')
        elif ft == 'singular' and entry_pos == 'verb':
            mapped_forms.append('third_person_singular')

        for mf in mapped_forms:
            val = (final_ent_id, mf, inflected_form)
            if val not in seen_inflections:
                seen_inflections.add(val)
                cur_final.execute(
                    "INSERT INTO entry_inflections (entry_id, form_type, inflected_form) VALUES (?, ?, ?)",
                    val
                )
                inflections_added += 1

    conn_final.commit()
    print(f"Migrated {inflections_added} entry inflections.")

    # final sense mapping: unified_sense_id -> final_sense_id
    unified_sense_id_to_final_id = {}
    # cambridge_sense_id -> list of final_sense_ids (split senses)
    cam_sense_id_to_final_ids = {}

    print("\n--- PHASE 4: MIGRATING CONSOLIDATED SENSES & ALTERNATIVE DEFINITIONS ---")
    
    print("Loading Open English WordNet 2024...")
    oewn = wn.Wordnet('oewn:2024')

    # Pre-fetch WordNet links grouped by unified_sense_id
    cur_align.execute("SELECT unified_sense_id, wordnet_id FROM unified_sense_wordnet_links")
    links_rows = cur_align.fetchall()
    uni_id_to_wn_ids = {}
    for uni_sid, wn_id in links_rows:
        if uni_sid not in uni_id_to_wn_ids:
            uni_id_to_wn_ids[uni_sid] = []
        uni_id_to_wn_ids[uni_sid].append(wn_id)

    cur_align.execute("SELECT id, word, pos, definition, cambridge_sense_id FROM unified_senses")
    uni_senses = cur_align.fetchall()

    # Pre-fetch Cambridge sense metadata (including original definition) to avoid N+1 queries
    cur_cam.execute("SELECT id, cefr_level, guideword, grammar, domain, labels, definition FROM senses")
    cam_sense_meta = {row[0]: row[1:] for row in cur_cam.fetchall()}

    alt_defs_added = 0

    for uni_id, word, pos, definition, cam_sid in tqdm(uni_senses, desc="Senses migration"):
        # Resolve target entry_id
        final_w_id = word_display_to_final_id[word.lower().strip()]
        pos_enum = map_pos_char_to_enum(pos) if len(pos) == 1 else map_pos_to_enum(pos)
        final_ent_id = final_entries_registry.get((final_w_id, pos_enum))
        if not final_ent_id:
            # Fallback to other POS
            final_ent_id = final_entries_registry.get((final_w_id, 'other'))
            if not final_ent_id:
                # If still not found, default to first entry of word
                cur_final.execute("SELECT id FROM entries WHERE word_id = ? LIMIT 1", (final_w_id,))
                res = cur_final.fetchone()
                if res:
                    final_ent_id = res[0]
                else:
                    # Create default entry
                    cur_final.execute(
                        "INSERT INTO entries (word_id, pos) VALUES (?, 'other')",
                        (final_w_id,)
                    )
                    final_ent_id = cur_final.lastrowid
                    final_entries_registry[(final_w_id, 'other')] = final_ent_id

        # Retrieve metadata
        orig_cam_def = None
        if cam_sid is not None and cam_sid in cam_sense_meta:
            cefr, guideword, grammar, domain, labels, orig_cam_def = cam_sense_meta[cam_sid]
            source = 'cambridge'
        else:
            cefr, guideword, grammar, domain, labels = None, None, None, None, '[]'
            source = 'wordnet'

        # Validate cefr_level against enum
        valid_cefr = cefr if cefr in ('A1', 'A2', 'B1', 'B2', 'C1', 'C2') else None

        cur_final.execute(
            "INSERT INTO senses (entry_id, definition, cefr_level, guideword, grammar, domain, labels) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (final_ent_id, definition, valid_cefr, guideword, grammar, domain, labels)
        )
        final_s_id = cur_final.lastrowid
        unified_sense_id_to_final_id[uni_id] = final_s_id
        
        if cam_sid is not None:
            if cam_sid not in cam_sense_id_to_final_ids:
                cam_sense_id_to_final_ids[cam_sid] = []
            cam_sense_id_to_final_ids[cam_sid].append(final_s_id)

        # 1. Insert original Cambridge definition
        if orig_cam_def:
            cur_final.execute(
                "INSERT INTO alternative_definitions (sense_id, definition, source, wordnet_id) VALUES (?, ?, 'cambridge', NULL)",
                (final_s_id, orig_cam_def)
            )
            alt_defs_added += 1

        # 2. Insert WordNet definitions
        wn_ids = uni_id_to_wn_ids.get(uni_id, [])
        for wn_id in wn_ids:
            try:
                wn_def = oewn.synset(wn_id).definition()
            except Exception:
                wn_def = ""
            if wn_def:
                cur_final.execute(
                    "INSERT INTO alternative_definitions (sense_id, definition, source, wordnet_id) VALUES (?, ?, 'wordnet', ?)",
                    (final_s_id, wn_def, wn_id)
                )
                alt_defs_added += 1

    conn_final.commit()
    print(f"Migrated {len(uni_senses)} unified senses and {alt_defs_added} alternative definitions.")

    # Cache target sense POS and word display forms to optimize synonyms filter
    cur_final.execute("""
        SELECT s.id, w.word, e.pos 
        FROM senses s 
        JOIN entries e ON s.entry_id = e.id 
        JOIN words w ON e.word_id = w.id
    """)
    senses_cache = {row[0]: (row[1], row[2]) for row in cur_final.fetchall()}

    def should_keep_synonym(src_pos, tgt_pos):
        if src_pos == 'other' or tgt_pos == 'other':
            return True
        return src_pos == tgt_pos

    print("\n--- PHASE 5: MIGRATING SYNONYMS & ANTONYMS ---")
    
    # Part A: Cambridge synonyms and antonyms
    cur_cam.execute("SELECT sense_id, synonym, target_sense_id, is_antonym FROM sense_synonyms")
    cam_syns = cur_cam.fetchall()
    
    syn_ant_inserted = 0
    
    for cam_sid, synonym, cam_target_sid, is_antonym in tqdm(cam_syns, desc="Cambridge synonyms"):
        final_src_sids = cam_sense_id_to_final_ids.get(cam_sid, [])
        for src_sid in final_src_sids:
            src_word, src_pos = senses_cache[src_sid]
            
            # Map target sense ID
            target_sid = None
            if cam_target_sid is not None:
                final_tgt_sids = cam_sense_id_to_final_ids.get(cam_target_sid, [])
                if final_tgt_sids:
                    # Pick the first matching pos or just the first target sense
                    chosen_tgt = None
                    for t_sid in final_tgt_sids:
                        _, t_pos = senses_cache[t_sid]
                        if should_keep_synonym(src_pos, t_pos):
                            chosen_tgt = t_sid
                            break
                    target_sid = chosen_tgt if chosen_tgt is not None else final_tgt_sids[0]
            
            # POS Filter check
            keep = True
            if target_sid is not None:
                _, tgt_pos = senses_cache[target_sid]
                if not should_keep_synonym(src_pos, tgt_pos):
                    keep = False
                    
            if keep:
                cur_final.execute(
                    "INSERT INTO synonyms_antonyms (sense_id, target_word, target_sense_id, is_antonym, source) VALUES (?, ?, ?, ?, 'cambridge')",
                    (src_sid, synonym, target_sid, is_antonym, )
                )
                syn_ant_inserted += 1

    print(f"Migrated {syn_ant_inserted} synonyms/antonyms from Cambridge.")

    # Part B: WordNet Synonyms & Antonyms
    print("Resolving WordNet synonym/antonym links...")

    # Load mappings from unified_sense_wordnet_links
    cur_align.execute("SELECT unified_sense_id, wordnet_id FROM unified_sense_wordnet_links")
    links_rows = cur_align.fetchall()
    
    wn_id_to_final_senses = {} # wordnet_id -> list of final_sense_ids
    final_sense_to_wn_ids = {} # final_sense_id -> list of wordnet_ids
    
    for uni_sid, wn_id in links_rows:
        final_sid = unified_sense_id_to_final_id.get(uni_sid)
        if final_sid:
            if wn_id not in wn_id_to_final_senses:
                wn_id_to_final_senses[wn_id] = []
            wn_id_to_final_senses[wn_id].append(final_sid)
            
            if final_sid not in final_sense_to_wn_ids:
                final_sense_to_wn_ids[final_sid] = []
            final_sense_to_wn_ids[final_sid].append(wn_id)

    wn_synonyms_inserted = 0
    wn_antonyms_inserted = 0
    
    # Process WordNet synonyms (lemmas within same synset)
    for wn_id, final_sids in tqdm(wn_id_to_final_senses.items(), desc="WordNet synonyms"):
        try:
            ss = oewn.synset(wn_id)
            lemmas = [w.lemma() for w in ss.words()]
        except Exception:
            continue
            
        for src_sid in final_sids:
            src_word, src_pos = senses_cache[src_sid]
            for target_word in lemmas:
                if target_word.lower().strip() != src_word.lower().strip():
                    # Resolve target_sense_id: does there exist a final sense for target_word linked to same synset?
                    target_sid = None
                    sibling_sids = wn_id_to_final_senses.get(wn_id, [])
                    for sib_sid in sibling_sids:
                        sib_word, _ = senses_cache[sib_sid]
                        if sib_word.lower().strip() == target_word.lower().strip():
                            target_sid = sib_sid
                            break
                            
                    cur_final.execute(
                        "INSERT INTO synonyms_antonyms (sense_id, target_word, target_sense_id, is_antonym, source) VALUES (?, ?, ?, 0, 'wordnet')",
                        (src_sid, target_word, target_sid)
                    )
                    wn_synonyms_inserted += 1

    # Process WordNet antonyms (sense-level antonym relationships)
    processed_antonym_pairs = set()
    for final_sid, wn_ids in tqdm(final_sense_to_wn_ids.items(), desc="WordNet antonyms"):
        src_word, src_pos = senses_cache[final_sid]
        for wn_id in wn_ids:
            try:
                ss = oewn.synset(wn_id)
            except Exception:
                continue
            for s in ss.senses():
                if s.word().lemma().lower().strip() == src_word.lower().strip():
                    # Check antonyms
                    if 'antonym' in s.relations():
                        for target_sense in s.relations()['antonym']:
                            target_lemma = target_sense.word().lemma()
                            target_wn_id = target_sense.synset().id
                            
                            # Find target final sense
                            target_sid = None
                            sib_tgt_sids = wn_id_to_final_senses.get(target_wn_id, [])
                            for sib_sid in sib_tgt_sids:
                                sib_word, sib_pos = senses_cache[sib_sid]
                                if sib_word.lower().strip() == target_lemma.lower().strip():
                                    target_sid = sib_sid
                                    break
                            
                            pair_key = (min(final_sid, target_sid) if target_sid else final_sid, max(final_sid, target_sid) if target_sid else target_lemma)
                            if pair_key not in processed_antonym_pairs:
                                processed_antonym_pairs.add(pair_key)
                                cur_final.execute(
                                    "INSERT INTO synonyms_antonyms (sense_id, target_word, target_sense_id, is_antonym, source) VALUES (?, ?, ?, 1, 'wordnet')",
                                    (final_sid, target_lemma, target_sid)
                                )
                                wn_antonyms_inserted += 1

    conn_final.commit()
    print(f"Migrated {wn_synonyms_inserted} synonyms and {wn_antonyms_inserted} antonyms from WordNet.")

    print("\n--- PHASE 6: MIGRATING EXAMPLES ---")
    
    # Part A: Cambridge Examples
    cur_cam.execute("SELECT sense_id, example, collocation FROM examples")
    cam_examples = cur_cam.fetchall()
    
    examples_inserted = 0
    for cam_sid, example, collocation in tqdm(cam_examples, desc="Cambridge examples"):
        final_sids = cam_sense_id_to_final_ids.get(cam_sid, [])
        for src_sid in final_sids:
            cur_final.execute(
                "INSERT INTO examples (sense_id, example_text, collocation, source) VALUES (?, ?, ?, 'cambridge')",
                (src_sid, example, collocation)
            )
            examples_inserted += 1

    # Part B: WordNet Examples
    wn_examples_inserted = 0
    for final_sid, wn_ids in tqdm(final_sense_to_wn_ids.items(), desc="WordNet examples"):
        for wn_id in wn_ids:
            try:
                ss = oewn.synset(wn_id)
                examples_list = ss.examples()
            except Exception:
                continue
            for ex in examples_list:
                cur_final.execute(
                    "INSERT INTO examples (sense_id, example_text, collocation, source) VALUES (?, ?, NULL, 'wordnet')",
                    (final_sid, ex)
                )
                wn_examples_inserted += 1

    conn_final.commit()
    print(f"Migrated {examples_inserted} Cambridge examples and {wn_examples_inserted} WordNet examples.")

    print("\n--- PHASE 7: MIGRATING COLLOCATIONS ---")
    cur_cam.execute("SELECT word_id, collocation, example FROM collocations")
    cam_collocations = cur_cam.fetchall()

    collocations_inserted = 0
    for cam_w_id, collocation, example in cam_collocations:
        final_w_id = cam_word_id_to_final_id.get(cam_w_id)
        if final_w_id:
            cur_final.execute(
                "INSERT INTO collocations (word_id, collocation, example) VALUES (?, ?, ?)",
                (final_w_id, collocation, example)
            )
            collocations_inserted += 1

    conn_final.commit()
    print(f"Migrated {collocations_inserted} collocations.")

    print("\n--- PHASE 8: MIGRATING TOPICS ---")
    # Topics
    cur_cam.execute("SELECT id, slug, title FROM topics")
    cam_topics = cur_cam.fetchall()
    for t_id, slug, title in cam_topics:
        cur_final.execute(
            "INSERT INTO topics (id, slug, title) VALUES (?, ?, ?)",
            (t_id, slug, title)
        )
    print(f"Migrated {len(cam_topics)} topics.")

    # Word Topics to Sense Topics
    cur_cam.execute("SELECT word_id, topic_id FROM word_topics")
    cam_word_topics = cur_cam.fetchall()

    sense_topics_inserted = 0
    seen_sense_topics = set()
    
    for cam_w_id, topic_id in cam_word_topics:
        final_w_id = cam_word_id_to_final_id.get(cam_w_id)
        if not final_w_id:
            continue
        # Find all senses of this word
        cur_final.execute("""
            SELECT s.id FROM senses s 
            JOIN entries e ON s.entry_id = e.id 
            WHERE e.word_id = ?
        """, (final_w_id,))
        for row in cur_final.fetchall():
            final_sid = row[0]
            val = (final_sid, topic_id)
            if val not in seen_sense_topics:
                seen_sense_topics.add(val)
                cur_final.execute(
                    "INSERT INTO sense_topics (sense_id, topic_id) VALUES (?, ?)",
                    val
                )
                sense_topics_inserted += 1

    conn_final.commit()
    print(f"Mapped {sense_topics_inserted} sense-topic associations.")

    print("\n--- PHASE 9: MIGRATING SEMANTIC RELATIONSHIPS ---")
    
    # Mapping table from WordNet relation types to relation_type enum values
    wn_relation_mapping = {
        'hypernym': 'hypernym',
        'instance_hypernym': 'hypernym',
        'hyponym': 'hyponym',
        'instance_hyponym': 'hyponym',
        'mero_part': 'part_meronym',
        'holo_member': 'member_holonym',
        'causes': 'cause',
        'entails': 'entailment',
        'similar': 'similar_to',
        'attribute': 'attribute',
        'domain_topic': 'domain_topic',
        'has_domain_topic': 'domain_topic',
        'domain_region': 'domain_region',
        'has_domain_region': 'domain_region',
        'domain_usage': 'domain_usage',
        'has_domain_usage': 'domain_usage',
        
        # Sense-level mappings
        'derivation': 'derivationally_related',
        'pertainym': 'derivationally_related',
        'participle': 'derivationally_related'
    }

    semantic_relations_inserted = 0
    seen_semantic_relations = set()

    def add_semantic_relation(src_sid, tgt_sid, rel_type, rel_src):
        # Prevent self-relations
        if src_sid == tgt_sid:
            return
        key = (src_sid, tgt_sid, rel_type, rel_src)
        if key not in seen_semantic_relations:
            seen_semantic_relations.add(key)
            cur_final.execute(
                "INSERT INTO semantic_relations (source_sense_id, target_sense_id, relation_type, source) VALUES (?, ?, ?, ?)",
                (src_sid, tgt_sid, rel_type, rel_src)
            )
            nonlocal semantic_relations_inserted
            semantic_relations_inserted += 1

    # Part A & B: WordNet synset & sense relations
    for src_wn_id, src_sids in tqdm(wn_id_to_final_senses.items(), desc="WordNet graph relations"):
        try:
            ss = oewn.synset(src_wn_id)
        except Exception:
            continue
            
        # 1. Synset-level relations
        for rel_name, target_synsets in ss.relations().items():
            mapped_rel = wn_relation_mapping.get(rel_name)
            if mapped_rel:
                for tgt_ss in target_synsets:
                    tgt_wn_id = tgt_ss.id
                    tgt_sids = wn_id_to_final_senses.get(tgt_wn_id, [])
                    for src_sid in src_sids:
                        for tgt_sid in tgt_sids:
                            add_semantic_relation(src_sid, tgt_sid, mapped_rel, 'wordnet')
                            
        # 2. Sense-level relations
        src_word_to_sense = {s.word().lemma().lower().strip(): s for s in ss.senses()}
        for src_sid in src_sids:
            src_word, _ = senses_cache[src_sid]
            word_key = src_word.lower().strip()
            if word_key in src_word_to_sense:
                s = src_word_to_sense[word_key]
                for rel_name, target_senses in s.relations().items():
                    mapped_rel = wn_relation_mapping.get(rel_name)
                    if mapped_rel:
                        for t_sense in target_senses:
                            t_lemma = t_sense.word().lemma()
                            t_wn_id = t_sense.synset().id
                            # Find target final sense matching t_lemma and t_wn_id
                            sib_tgt_sids = wn_id_to_final_senses.get(t_wn_id, [])
                            for tgt_sid in sib_tgt_sids:
                                sib_word, _ = senses_cache[tgt_sid]
                                if sib_word.lower().strip() == t_lemma.lower().strip():
                                    add_semantic_relation(src_sid, tgt_sid, mapped_rel, 'wordnet')

    print(f"Inserted {semantic_relations_inserted} semantic relations from WordNet.")

    # Part C: ConceptNet Relations
    cur_align.execute("""
        SELECT m1.wordnet_id, r.relation, m2.wordnet_id
        FROM conceptnet_relations r
        JOIN conceptnet_sense_mappings m1 ON r.start_concept = m1.conceptnet_uri
        JOIN conceptnet_sense_mappings m2 ON r.end_concept = m2.conceptnet_uri
    """)
    conceptnet_rows = cur_align.fetchall()

    conceptnet_relation_mapping = {
        'Antonym': 'antonym',
        'CapableOf': 'capable_of',
        'Causes': 'causes',
        'HasProperty': 'has_property',
        'MadeOf': 'part_of',
        'PartOf': 'part_of',
        'Synonym': 'synonym',
        'UsedFor': 'used_for'
    }

    conceptnet_inserted = 0
    for src_wn_id, rel_name, tgt_wn_id in tqdm(conceptnet_rows, desc="ConceptNet graph relations"):
        mapped_rel = conceptnet_relation_mapping.get(rel_name)
        if mapped_rel:
            src_sids = wn_id_to_final_senses.get(src_wn_id, [])
            tgt_sids = wn_id_to_final_senses.get(tgt_wn_id, [])
            for src_sid in src_sids:
                for tgt_sid in tgt_sids:
                    add_semantic_relation(src_sid, tgt_sid, mapped_rel, 'conceptnet')
                    conceptnet_inserted += 1

    conn_final.commit()
    print(f"Inserted {conceptnet_inserted} semantic relations from ConceptNet.")

    # Clean up and optimize
    print("\nRunning database optimizations (VACUUM and ANALYZE)...")
    cur_final.execute("VACUUM")
    cur_final.execute("ANALYZE")
    conn_final.commit()

    # Close all connections
    conn_cam.close()
    conn_align.close()
    conn_final.close()

    print("\n" + "="*80)
    print("CONSOLIDATED DATABASE BUILD COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Final Database File: {final_db_path}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
