import sqlite3
import os
import json
import math
import heapq
import threading
import time
import queue
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import unicodedata

import wn
from src.utils.nlp import map_pos, resolve_alternatives, get_match_keys

from src.utils.db import fetch_cambridge_senses
from src.alignment.sense_aligner import (
    build_alignment_prompt,
    call_alignment_llm,
    parse_alignment_response
)

db_write_queue = queue.Queue()

def db_writer_worker(db_path):
    """Worker thread that executes batch SQL updates to SQLite."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Initialize schema in data/dictionary_alignment.db
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS unified_senses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT NOT NULL,
            pos TEXT NOT NULL,
            definition TEXT NOT NULL,
            cambridge_sense_id INTEGER
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_unified_senses_cambridge_id ON unified_senses(cambridge_sense_id)")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS unified_sense_wordnet_links (
            unified_sense_id INTEGER NOT NULL,
            wordnet_id TEXT NOT NULL,
            PRIMARY KEY (unified_sense_id, wordnet_id),
            FOREIGN KEY (unified_sense_id) REFERENCES unified_senses(id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS word_pos_alignment_status (
            word TEXT NOT NULL,
            pos TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            PRIMARY KEY (word, pos)
        )
    """)
    # word_alternatives dropped from alignment database (fully maintained at sense-level in cambridge.db)
    conn.commit()
    
    print("Database writer worker started.")
    
    while True:
        item = db_write_queue.get()
        if item is None:
            db_write_queue.task_done()
            break
            
        word = item['word']
        pos_char = item['pos']
        unified_senses = item['unified_senses']
        
        try:
            # Transaction starts here
            # Clear old mappings for this (word, pos) to guarantee clean overwrite
            cursor.execute("SELECT id FROM unified_senses WHERE word = ? AND pos = ?", (word, pos_char))
            old_ids = [r[0] for r in cursor.fetchall()]
            if old_ids:
                placeholders = ",".join("?" for _ in old_ids)
                cursor.execute(f"DELETE FROM unified_sense_wordnet_links WHERE unified_sense_id IN ({placeholders})", old_ids)
                cursor.execute(f"DELETE FROM unified_senses WHERE id IN ({placeholders})", old_ids)
            
            # Write new unified senses
            for sense in unified_senses:
                definition = sense['definition']
                cam_ids = sense['cambridge_sense_ids']
                cam_id = cam_ids[0] if cam_ids else None
                wn_ids = sense['wordnet_synset_ids']
                
                cursor.execute(
                    "INSERT INTO unified_senses (word, pos, definition, cambridge_sense_id) VALUES (?, ?, ?, ?)",
                    (word, pos_char, definition, cam_id)
                )
                u_id = cursor.lastrowid
                
                for wn_id in wn_ids:
                    cursor.execute(
                        "INSERT INTO unified_sense_wordnet_links (unified_sense_id, wordnet_id) VALUES (?, ?)",
                        (u_id, wn_id)
                    )
            
            # Update status
            cursor.execute(
                "UPDATE word_pos_alignment_status SET status = 'aligned' WHERE word = ? AND pos = ?",
                (word, pos_char)
            )
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            print(f"Error executing DB update for {word}_{pos_char}: {e}")
            
        db_write_queue.task_done()
        
    conn.close()
    print("Database writer worker stopped.")


def thread_call_llm(model_id, api_key, api_base, reasoning_effort, system_instruction, user_prompt, batch_lookups):
    """Wrapper to make parallel LLM API calls and parse response."""
    res_dict = call_alignment_llm(model_id, system_instruction, user_prompt, api_key, api_base, reasoning_effort)
    if res_dict.get("error"):
        return None, 0, 0, res_dict["error"]
        
    batch_mappings = parse_alignment_response(res_dict["content"], batch_lookups)
    if batch_mappings is None:
        return None, res_dict["prompt_tokens"], res_dict["completion_tokens"], "Failed to parse JSON alignment response"
    return batch_mappings, res_dict["prompt_tokens"], res_dict["completion_tokens"], None


def align_database(
    model_id: str,
    reasoning_effort: str = "none",
    limit: int = None,
    output_path: str = "data/word_senses_alignment.json",
    workers: int = 16,
    batch_size: int = 8,
    difficulty_method: str = "token",
    db_path: str = "data/dictionary_alignment.db",
    cambridge_db_path: str = "data/cambridge.db"
):
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_BASE_URL")

    if not api_key or not api_base:
        print("Error: OPENAI_API_KEY and OPENAI_BASE_URL must be set in the .env file.")
        return

    # Start the DB writer thread
    writer_thread = threading.Thread(target=db_writer_worker, args=(db_path,))
    writer_thread.daemon = True
    writer_thread.start()

    # Give the writer thread a brief moment to ensure schema tables are created
    time.sleep(0.5)

    # 1. Initialize tasks in the status table if it is empty
    conn_align = sqlite3.connect(db_path)
    cursor_align = conn_align.cursor()
    cursor_align.execute("SELECT COUNT(*) FROM word_pos_alignment_status")
    if cursor_align.fetchone()[0] == 0:
        print("Initializing status tracking table from Cambridge database...")
        conn_cam = sqlite3.connect(cambridge_db_path)
        cursor_cam = conn_cam.cursor()
        
        # Query all senses to filter out grammatical inflections and spelling variants
        cursor_cam.execute("""
            SELECT w.display_form, e.pos, s.phrase_title, s.id, s.definition
            FROM words w
            JOIN entries e ON w.id = e.word_id
            JOIN senses s ON e.id = s.entry_id
            WHERE e.dictionary_source = "Cambridge Advanced Learner's Dictionary"
            ORDER BY w.display_form, e.pos, s.id
        """)
        rows = cursor_cam.fetchall()
        
        # Group senses by word_pos
        from collections import defaultdict
        word_pos_senses = defaultdict(list)
        for word_display, pos, phrase_title, s_id, defn in rows:
            word = phrase_title if (phrase_title and phrase_title.strip()) else word_display
            pos_clean = pos.lower().strip() if pos else 'phrase'
            if pos_clean in ('idiom', 'phrasal verb', 'phrase', 'collocation'):
                pos_char = pos_clean
            else:
                pos_char = map_pos(pos_clean)
                
            if pos_char:
                word_pos_senses[(word, pos_char)].append((s_id, defn))
                
        # Insert all remaining tasks (redirect/inflection senses have been removed at crawl time)
        batch_insert = []
        for (word, pos_char) in word_pos_senses.keys():
            batch_insert.append((word, pos_char, "pending"))
                
        cursor_align.executemany(
            "INSERT OR IGNORE INTO word_pos_alignment_status (word, pos, status) VALUES (?, ?, ?)",
            batch_insert
        )
        
        conn_cam.close()
        conn_align.commit()
        print(f"Tracking status initialized for {len(batch_insert)} word-POS tasks.")

    # 2. Query pending tasks from status table in deterministic order
    cursor_align.execute("SELECT word, pos FROM word_pos_alignment_status WHERE status = 'pending' ORDER BY word, pos")
    pending_tasks = cursor_align.fetchall()
    pending_set = {(word, pos) for word, pos in pending_tasks}
    
    # First count total CALD senses in cambridge.db for reporting
    conn_cam = sqlite3.connect(cambridge_db_path)
    cursor_cam = conn_cam.cursor()
    cursor_cam.execute("""
        SELECT COUNT(*) FROM senses s
        JOIN entries e ON e.id = s.entry_id
        WHERE e.dictionary_source = "Cambridge Advanced Learner's Dictionary"
    """)
    total_cald_senses = cursor_cam.fetchone()[0]

    # Fetch all Cambridge senses in deterministic order to group them in memory
    print("Reading Cambridge senses in memory...")
    cursor_cam.execute("""
        SELECT w.display_form, e.pos, s.phrase_title, s.id, s.definition
        FROM words w
        JOIN entries e ON w.id = e.word_id
        JOIN senses s ON e.id = s.entry_id
        WHERE e.dictionary_source = "Cambridge Advanced Learner's Dictionary"
        ORDER BY w.display_form, e.pos, s.id
    """)
    rows = cursor_cam.fetchall()
    conn_cam.close()
    
    # Senses grouping for memory loading

    from collections import defaultdict
    cam_data = defaultdict(list)
    for word_display, pos, phrase_title, s_id, definition in rows:
        word = phrase_title if (phrase_title and phrase_title.strip()) else word_display
        pos_clean = pos.lower().strip() if pos else 'phrase'
        if pos_clean in ('idiom', 'phrasal verb', 'phrase', 'collocation'):
            pos_char = pos_clean
        else:
            pos_char = map_pos(pos_clean)
            
        if pos_char:
            if (word, pos_char) in pending_set:
                cam_data[(word, pos_char)].append((s_id, pos, definition))
            
    tasks_to_process = sorted(list(cam_data.keys()))
    
    print(f"Total Cambridge senses: {total_cald_senses}.")
    print(f"Pending tasks remaining: {len(pending_set)}.")
    
    if limit is not None:
        tasks_to_process = tasks_to_process[:limit]
        print(f"Limiting execution to the first {limit} remaining tasks.")
        
    if not tasks_to_process:
        print("All word-POS tasks are already fully aligned! Exiting.")
        cursor_align.close()
        conn_align.close()
        db_write_queue.put(None)
        writer_thread.join()
        return

    cursor_align.close()
    conn_align.close()

    # 3. Load WordNet synsets in memory for tasks to process using normalized keys
    print("Loading WordNet and pre-matching synsets in memory...")
    oewn = wn.Wordnet('oewn:2024')
    
    # Pre-calculate normalized keys for Cambridge tasks to filter WordNet quickly
    cam_task_normalized_keys = {}
    for word, pos_char in tasks_to_process:
        cam_task_normalized_keys[(word, pos_char)] = get_match_keys(word)
        
    search_candidates_set = set()
    for cands in cam_task_normalized_keys.values():
        search_candidates_set.update(cands)
        
    from collections import defaultdict
    wn_data = defaultdict(list)  # Key: (normalized_key, pos_char) -> list of synset dicts
    for w in oewn.words():
        lemma = w.lemma()
        pos = w.pos
        pos_char = 'a' if pos == 's' else pos
        
        wn_cands = get_match_keys(lemma)
        matching_cands = [c for c in wn_cands if c in search_candidates_set]
        if matching_cands:
            seen_synsets = set()
            synsets_data = []
            for ss in w.synsets():
                if ss.id not in seen_synsets:
                    seen_synsets.add(ss.id)
                    synsets_data.append({
                        'id': ss.id,
                        'pos': ss.pos,
                        'definition': ss.definition()
                    })
            if synsets_data:
                for cand in matching_cands:
                    wn_data[(cand, pos_char)].extend(synsets_data)

    # 4. Pipeline Pre-Processing & POS-filtering (Phase 1)
    print("Running initial POS mismatch checks...")
    all_ambiguous_tasks = []
    
    for word, pos_char in tasks_to_process:
        cam_senses = cam_data[(word, pos_char)]
        
        # Collect WordNet synsets matching any candidate key of this Cambridge word
        candidates = cam_task_normalized_keys[(word, pos_char)]
        wn_list_for_pos = []
        seen_wn_ids = set()
        for cand in candidates:
            if pos_char in ('idiom', 'phrase', 'collocation'):
                for p in ('n', 'v', 'a', 'r'):
                    for item in wn_data.get((cand, p), []):
                        if item['id'] not in seen_wn_ids:
                            seen_wn_ids.add(item['id'])
                            wn_list_for_pos.append(item)
            else:
                for item in wn_data.get((cand, pos_char), []):
                    if item['id'] not in seen_wn_ids:
                        seen_wn_ids.add(item['id'])
                        wn_list_for_pos.append(item)
        
        # POS Mismatch Bypass
        if not wn_list_for_pos:
            # Output each Cambridge sense as its own unified sense with no WordNet mapping
            bypassed_senses = []
            for s_id, _, definition in cam_senses:
                bypassed_senses.append({
                    'definition': definition,
                    'cambridge_sense_ids': [s_id],
                    'wordnet_synset_ids': []
                })
            db_write_queue.put({
                'word': word,
                'pos': pos_char,
                'unified_senses': bypassed_senses
            })
            continue
            
        # Push to parallel LLM alignment
        all_ambiguous_tasks.append({
            'task_id': f"{word}_{pos_char}",
            'word': word,
            'pos': pos_char,
            'cam_senses': cam_senses,
            'wn_synsets': wn_list_for_pos
        })

    # 5. Dynamic Batching and Load Balancing (Phase 2)
    if all_ambiguous_tasks:
        if difficulty_method == "token":
            import tiktoken
            try:
                enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                enc = tiktoken.encoding_for_model("gpt-4")
            
            print("Pre-calculating token sizes for dynamic batching...")
            task_tokens = {}
            for t in tqdm(all_ambiguous_tasks, desc="Calculating tokens"):
                word = t["word"]
                task_id = t["task_id"]
                cam_text = "".join(f"[{idx+1}] {defn}\n" for idx, (_, _, defn) in enumerate(t["cam_senses"]))
                wn_text = ""
                for idx, ss in enumerate(t["wn_synsets"]):
                    ss_def = ss.definition() if not isinstance(ss, dict) else ss["definition"]
                    wn_text += f"[{idx+1}] {ss_def}\n"
                
                task_text = f"  <task id=\"{task_id}\">\n    <target_word>{word}</target_word>\n    <cambridge_senses>\n{cam_text.strip()}\n    </cambridge_senses>\n    <wordnet_synsets>\n{wn_text.strip()}\n    </wordnet_synsets>\n  </task>\n\n"
                task_tokens[task_id] = len(enc.encode(task_text))
            
            diff_fn = lambda t: task_tokens[t["task_id"]]
        elif difficulty_method == "add":
            diff_fn = lambda t: len(t["cam_senses"]) + len(t["wn_synsets"])
        else:
            diff_fn = lambda t: len(t["cam_senses"]) * len(t["wn_synsets"])
            
        N = len(all_ambiguous_tasks)
        M = math.ceil(N / batch_size)
        
        all_ambiguous_tasks.sort(
            key=lambda t: (diff_fn(t), t["task_id"]),
            reverse=True
        )
        
        heap = [(0, i) for i in range(M)]
        heapq.heapify(heap)
        
        chunks = [[] for _ in range(M)]
        for task in all_ambiguous_tasks:
            curr_weight, b_idx = heapq.heappop(heap)
            chunks[b_idx].append(task)
            diff = diff_fn(task)
            heapq.heappush(heap, (curr_weight + diff, b_idx))
            
        for chunk in chunks:
            chunk.sort(key=lambda t: t["task_id"])
            
        import numpy as np
        chunk_sizes = [len(c) for c in chunks]
        chunk_tokens = []
        
        for chunk in chunks:
            total_t = 0
            for t in chunk:
                if difficulty_method == "token":
                    total_t += task_tokens[t["task_id"]]
                else:
                    word = t["word"]
                    task_id = t["task_id"]
                    cam_text = "".join(f"[{idx+1}] {defn}\n" for idx, (_, _, defn) in enumerate(t["cam_senses"]))
                    wn_text = ""
                    for idx, ss in enumerate(t["wn_synsets"]):
                        ss_def = ss.definition() if not isinstance(ss, dict) else ss["definition"]
                        wn_text += f"[{idx+1}] {ss_def}\n"
                    task_text = f"  <task id=\"{task_id}\">\n    <target_word>{word}</target_word>\n    <cambridge_senses>\n{cam_text.strip()}\n    </cambridge_senses>\n    <wordnet_synsets>\n{wn_text.strip()}\n    </wordnet_synsets>\n  </task>\n\n"
                    total_t += len(enc.encode(task_text))
            chunk_tokens.append(total_t)
            
        print("\n" + "="*60)
        print("DYNAMIC BATCHING AND LOAD BALANCING REPORT")
        print("="*60)
        print(f"  - Total LLM tasks:           {N}")
        print(f"  - Total LLM requests (M):     {M}")
        print(f"  - Batch size (tasks/request): min={min(chunk_sizes)}, max={max(chunk_sizes)}, avg={np.mean(chunk_sizes):.2f}")
        print(f"  - Prompt tokens per request:  min={min(chunk_tokens)}, max={max(chunk_tokens)}, avg={np.mean(chunk_tokens):.1f}, std_dev={np.std(chunk_tokens):.2f}")
        print("="*60 + "\n")
        
        batches_data = []
        for chunk in chunks:
            sys_inst, user_pr, batch_lookups = build_alignment_prompt(chunk)
            batches_data.append({
                'chunk': chunk,
                'sys_inst': sys_inst,
                'user_pr': user_pr,
                'batch_lookups': batch_lookups
            })
            
        print(f"Resolving {N} tasks in parallel with {workers} workers...")
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for b_task in batches_data:
                f = executor.submit(
                    thread_call_llm,
                    model_id,
                    api_key,
                    api_base,
                    reasoning_effort,
                    b_task['sys_inst'],
                    b_task['user_pr'],
                    b_task['batch_lookups']
                )
                futures.append((f, b_task['chunk']))
                
            for fut, chunk in tqdm(futures, desc="Aligning Senses"):
                try:
                    batch_mappings, p_tokens, c_tokens, err = fut.result()
                    if err:
                        print(f"\nError resolving batch: {err}")
                        continue
                    
                    total_prompt_tokens += p_tokens
                    total_completion_tokens += c_tokens
                    
                    # Sort the batch mappings by task_id before writing to ensure deterministic queueing
                    for task_id in sorted(batch_mappings.keys()):
                        unified_list = batch_mappings[task_id]
                        word, pos_char = task_id.rsplit("_", 1)
                        db_write_queue.put({
                            'word': word,
                            'pos': pos_char,
                            'unified_senses': unified_list
                        })
                            
                except Exception as e:
                    print(f"\nThread execution error: {e}")
                    
        print("\n" + "="*60)
        print("ALIGNMENT COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"  - Total Prompt Tokens:     {total_prompt_tokens}")
        print(f"  - Total Completion Tokens: {total_completion_tokens}")
        print(f"  - Total Tokens:            {total_prompt_tokens + total_completion_tokens}")
        print("="*60 + "\n")
    else:
        print("No pending ambiguous tasks remaining for LLM resolution.")

    # Stop the DB writer thread
    db_write_queue.put(None)
    writer_thread.join()

    # Export final JSON mapping if output_path is provided
    if output_path:
        print(f"Exporting final database alignments to {output_path}...")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Order by word, pos, id to guarantee 100% deterministic JSON output
        cursor.execute("SELECT id, word, pos, definition, cambridge_sense_id FROM unified_senses ORDER BY word, pos, id")
        u_senses = cursor.fetchall()
        
        export_data = {}
        for u_id, word, pos, definition, cam_id in u_senses:
            # Fetch links
            cam_ids = [cam_id] if cam_id is not None else []
            
            cursor.execute("SELECT wordnet_id FROM unified_sense_wordnet_links WHERE unified_sense_id = ? ORDER BY wordnet_id", (u_id,))
            wn_ids = [r[0] for r in cursor.fetchall()]
            
            task_key = f"{word}_{pos}"
            if task_key not in export_data:
                export_data[task_key] = []
                
            export_data[task_key].append({
                "definition": definition,
                "cambridge_sense_ids": cam_ids,
                "wordnet_synset_ids": wn_ids
            })
            
        conn.close()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"Export completed! File saved at {output_path}")
