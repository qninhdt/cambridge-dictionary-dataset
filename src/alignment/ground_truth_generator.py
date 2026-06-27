import sqlite3
import wn
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.db import fetch_cambridge_senses
from src.utils.nlp import map_pos
from src.alignment.sense_aligner import (
    build_alignment_prompt, 
    call_alignment_llm, 
    parse_alignment_response
)

def thread_call_llm(model_id, api_key, api_base, reasoning_effort, system_instruction, user_prompt, batch_lookups):
    res_dict = call_alignment_llm(model_id, system_instruction, user_prompt, api_key, api_base, reasoning_effort)
    if res_dict.get("error"):
        return None, 0, 0, res_dict["error"]
        
    batch_mappings = parse_alignment_response(res_dict["content"], batch_lookups)
    return batch_mappings, res_dict["prompt_tokens"], res_dict["completion_tokens"], None

def generate_ground_truth(
    model_id: str,
    reasoning_effort: str = "medium",
    limit: int = None,
    output_path: str = "data/ground_truth.json",
    workers: int = 16,
    batch_size: int = 1
):
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_BASE_URL")

    if not api_key or not api_base:
        print("Error: OPENAI_API_KEY and OPENAI_BASE_URL must be set in the .env file.")
        return

    custom_words_path = "data/representative_words.json"
    conn = sqlite3.connect("data/cambridge.db")
    cursor = conn.cursor()
    
    if os.path.exists(custom_words_path):
        print(f"Loading custom representative words from {custom_words_path}...")
        with open(custom_words_path, "r", encoding="utf-8") as f:
            test_words = json.load(f)
        if limit is not None:
            test_words = test_words[:limit]
    else:
        limit_val = limit if limit is not None else 100
        from src.utils.db import find_polysemous_words
        print(f"Dynamically selecting {limit_val} polysemous words with >= 3 senses in CALD and present in WordNet...")
        test_words = find_polysemous_words(cursor, limit=limit_val)
        
    print(f"Selected {len(test_words)} words for ground truth generation.")

    oewn = wn.Wordnet('oewn:2024')
    print("Querying databases and preparing tasks...")
    
    raw_tasks = []
    ground_truth = {}
    
    for word in test_words:
        cam_senses = fetch_cambridge_senses(cursor, word)
        wn_synsets = oewn.synsets(word)

        if not cam_senses or not wn_synsets:
            print(f"  [Warning] Skipping '{word}' due to missing senses in one of the datasets.")
            continue
            
        # Group Cambridge senses by POS mapping: 'n', 'v', 'a', 'r'
        cam_by_pos = {}
        for s_id, pos, definition in cam_senses:
            pos_char = map_pos(pos)
            if not pos_char:
                continue
            if pos_char not in cam_by_pos:
                cam_by_pos[pos_char] = []
            cam_by_pos[pos_char].append((s_id, pos, definition))
            
        # Group WordNet synsets by POS mapping
        wn_by_pos = {}
        for ss in wn_synsets:
            pos_char = ss.pos
            if pos_char == 's':  # Map satellite adjectives to 'a'
                pos_char = 'a'
            if pos_char not in wn_by_pos:
                wn_by_pos[pos_char] = []
            wn_by_pos[pos_char].append(ss)
            
        # Create separate tasks for each POS group
        for pos_char, cam_list in cam_by_pos.items():
            wn_list_for_pos = wn_by_pos.get(pos_char, [])
            t_id = f"{word}_{pos_char}"
            
            if not wn_list_for_pos:
                # Pre-fill bypassed N-0 senses
                ground_truth[t_id] = []
                for s_id, _, definition in cam_list:
                    ground_truth[t_id].append({
                        "definition": definition,
                        "cambridge_sense_ids": [s_id],
                        "wordnet_synset_ids": []
                    })
                continue
                
            raw_tasks.append({
                'task_id': t_id,
                'word': word,
                'pos': pos_char,
                'cam_senses': cam_list,
                'wn_synsets': wn_list_for_pos
            })
        
    conn.close()
    
    print(f"\nModel: {model_id}")
    print(f"Batch Size: {batch_size}")
    
    total_prompt_tokens = 0
    total_completion_tokens = 0
    errors_occurred = []

    from tqdm import tqdm
    
    # Prepare batches
    batches_data = []
    for i in range(0, len(raw_tasks), batch_size):
        chunk = raw_tasks[i:i+batch_size]
        sys_inst, user_pr, batch_lookups = build_alignment_prompt(chunk)
        batches_data.append({
            'chunk': chunk,
            'sys_inst': sys_inst,
            'user_pr': user_pr,
            'batch_lookups': batch_lookups
        })
        
    print(f"Generating ground truth (chunk size: {batch_size}) in parallel with {workers} workers...")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
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
            futures[f] = b_task['chunk']
            
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating GT"):
            chunk = futures[fut]
            chunk_label = ", ".join([f"{t['word']}_{t['pos']}" for t in chunk])
            try:
                batch_mappings, p_tokens, c_tokens, err = fut.result()
                total_prompt_tokens += p_tokens
                total_completion_tokens += c_tokens

                if err:
                    errors_occurred.append((chunk_label, err))
                elif batch_mappings:
                    for task in chunk:
                        t_id = task['task_id']
                        mappings = batch_mappings.get(t_id, [])
                        
                        if mappings:
                            ground_truth[t_id] = mappings
                        else:
                            # Backfill
                            ground_truth[t_id] = []
                            for s_id, _, definition in task['cam_senses']:
                                ground_truth[t_id].append({
                                    "definition": definition,
                                    "cambridge_sense_ids": [s_id],
                                    "wordnet_synset_ids": []
                                })
                else:
                    errors_occurred.append((chunk_label, "Empty mappings received"))
            except Exception as e:
                errors_occurred.append((chunk_label, str(e)))

    print("\n========================================================")
    print("TOKEN USAGE ANALYSIS")
    print("========================================================")
    print(f"Prompt (Input) Tokens:      {total_prompt_tokens}")
    print(f"Completion (Output) Tokens:  {total_completion_tokens}")
    print(f"Total Tokens:               {total_prompt_tokens + total_completion_tokens}")
    print("========================================================")

    if errors_occurred:
        print("\n[WARNING] Some requests failed. The analysis above reflects successful calls only.")
        for label, err in errors_occurred:
            print(f"  - '{label}': {err}")

    if ground_truth:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ground_truth, f, indent=2, ensure_ascii=False)
        print(f"\nGround truth successfully saved to: {output_path}")
