import sqlite3
import wn
import os
import json
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from src.utils.nlp import map_pos, extract_int
from src.utils.db import fetch_cambridge_senses
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


def calculate_clustering_metrics(gt_senses, pred_senses):
    """
    Calculate Precision, Recall, and F1 based on Jaccard Index of unified sense clusters.
    """
    if not gt_senses and not pred_senses:
        return 1.0, 1.0, 1.0
    if not gt_senses or not pred_senses:
        return 0.0, 0.0, 0.0
        
    # Precision: average best Jaccard match for each predicted cluster in GT
    p_scores = []
    for pred in pred_senses:
        p_cam = set(pred.get("cambridge_sense_ids", []))
        p_wn = set(pred.get("wordnet_synset_ids", []))
        
        max_jaccard = 0.0
        for gt in gt_senses:
            g_cam = set(gt.get("cambridge_sense_ids", []))
            g_wn = set(gt.get("wordnet_synset_ids", []))
            
            intersection = len(p_cam.intersection(g_cam)) + len(p_wn.intersection(g_wn))
            union = len(p_cam.union(g_cam)) + len(p_wn.union(g_wn))
            
            jaccard = intersection / union if union > 0 else 0.0
            if jaccard > max_jaccard:
                max_jaccard = jaccard
        p_scores.append(max_jaccard)
        
    # Recall: average best Jaccard match for each GT cluster in prediction
    r_scores = []
    for gt in gt_senses:
        g_cam = set(gt.get("cambridge_sense_ids", []))
        g_wn = set(gt.get("wordnet_synset_ids", []))
        
        max_jaccard = 0.0
        for pred in pred_senses:
            p_cam = set(pred.get("cambridge_sense_ids", []))
            p_wn = set(pred.get("wordnet_synset_ids", []))
            
            intersection = len(p_cam.intersection(g_cam)) + len(p_wn.intersection(g_wn))
            union = len(p_cam.union(g_cam)) + len(p_wn.union(g_wn))
            
            jaccard = intersection / union if union > 0 else 0.0
            if jaccard > max_jaccard:
                max_jaccard = jaccard
        r_scores.append(max_jaccard)
        
    precision = sum(p_scores) / len(p_scores) if p_scores else 0.0
    recall = sum(r_scores) / len(r_scores) if r_scores else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0.0 else 0.0
    
    return precision, recall, f1


def evaluate_hybrid_mapping(
    model_id: str,
    reasoning_effort: str = "medium",
    limit: int = None,
    gt_path: str = "data/ground_truth.json",
    workers: int = 16,
    use_reranker: bool = False,         # Left for backward compatibility, ignored
    accept_threshold: float = 0.35,     # Left for backward compatibility, ignored
    reject_threshold: float = 0.05,     # Left for backward compatibility, ignored
    batch_size: int = 1,
    difficulty_method: str = "token"
):
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_BASE_URL")

    if not api_key or not api_base:
        print("Error: OPENAI_API_KEY and OPENAI_BASE_URL must be set in the .env file.")
        return

    # Load Ground Truth
    if not os.path.exists(gt_path):
        print(f"Error: Ground truth file not found at: {gt_path}")
        return
        
    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)
        
    test_task_ids = list(ground_truth.keys())
    if limit is not None:
        test_task_ids = test_task_ids[:limit]
        print(f"Limiting evaluation to the first {limit} tasks.")
        
    print(f"Loaded ground truth containing {len(test_task_ids)} tasks.")
    
    # Establish connection to Cambridge DB (read-only)
    conn = sqlite3.connect("data/cambridge.db")
    cursor = conn.cursor()
    oewn = wn.Wordnet('oewn:2024')
    
    raw_tasks = []
    predictions = {}
    
    # Step 1: Pre-process tasks and identify POS mismatches
    print("Reading definitions and running POS filters...")
    resolved_by_pos_mismatch = 0
    total_cam_senses = 0
    
    for t_id in test_task_ids:
        # t_id format: word_pos
        word, pos_char = t_id.rsplit("_", 1)
        
        cam_senses = fetch_cambridge_senses(cursor, word)
        wn_synsets = oewn.synsets(word)
        
        # Filter senses by matching task POS
        cam_list = []
        for s_id, pos, definition in cam_senses:
            p_char = map_pos(pos)
            if p_char == pos_char:
                cam_list.append((s_id, pos, definition))
                total_cam_senses += 1
                
        wn_list = []
        for ss in wn_synsets:
            p_char = ss.pos
            if p_char == 's':
                p_char = 'a'
            if p_char == pos_char:
                wn_list.append({
                    'id': ss.id,
                    'pos': ss.pos,
                    'definition': ss.definition()
                })
                
        # POS Mismatch Bypass
        if not wn_list:
            predictions[t_id] = []
            for s_id, _, definition in cam_list:
                predictions[t_id].append({
                    "definition": definition,
                    "cambridge_sense_ids": [s_id],
                    "wordnet_synset_ids": []
                })
            resolved_by_pos_mismatch += len(cam_list)
            continue
            
        raw_tasks.append({
            'task_id': t_id,
            'word': word,
            'pos': pos_char,
            'cam_senses': cam_list,
            'wn_synsets': wn_list
        })
        
    conn.close()

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_reasoning_tokens = 0
    errors_occurred = []
    llm_calls_made = 0

    # Step 2: Call LLM in parallel batches
    if raw_tasks:
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
            
        print(f"Running evaluation batches (chunk size: {batch_size}) in parallel with {workers} workers...")
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
                
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Evaluating Senses"):
                chunk = futures[fut]
                chunk_label = ", ".join([f"{t['word']}_{t['pos']}" for t in chunk])
                try:
                    batch_mappings, p_tokens, c_tokens, err = fut.result()
                    total_prompt_tokens += p_tokens
                    total_completion_tokens += c_tokens
                    llm_calls_made += len(chunk)

                    if err:
                        errors_occurred.append((chunk_label, err))
                    elif batch_mappings:
                        for task in chunk:
                            t_id = task['task_id']
                            mappings = batch_mappings.get(t_id, [])
                            
                            if mappings:
                                predictions[t_id] = mappings
                            else:
                                # Backfill
                                predictions[t_id] = []
                                for s_id, _, definition in task['cam_senses']:
                                    predictions[t_id].append({
                                        "definition": definition,
                                        "cambridge_sense_ids": [s_id],
                                        "wordnet_synset_ids": []
                                    })
                    else:
                        errors_occurred.append((chunk_label, "Empty mappings received"))
                except Exception as e:
                    errors_occurred.append((chunk_label, str(e)))

    # Step 3: Compute clustering metrics
    prec_list = []
    rec_list = []
    f1_list = []
    
    detailed_samples = {}
    
    for t_id in test_task_ids:
        gt_senses = ground_truth.get(t_id, [])
        pred_senses = predictions.get(t_id, [])
        
        p, r, f1 = calculate_clustering_metrics(gt_senses, pred_senses)
        prec_list.append(p)
        rec_list.append(r)
        f1_list.append(f1)
        
        detailed_samples[t_id] = {
            "ground_truth": gt_senses,
            "prediction": pred_senses,
            "clustering_precision": p,
            "clustering_recall": r,
            "clustering_f1": f1
        }
        
    mean_precision = (sum(prec_list) / len(prec_list)) * 100 if prec_list else 0.0
    mean_recall = (sum(rec_list) / len(rec_list)) * 100 if rec_list else 0.0
    mean_f1 = (sum(f1_list) / len(f1_list)) * 100 if f1_list else 0.0
    
    llm_bypass_rate = (1 - (llm_calls_made / len(test_task_ids))) * 100 if len(test_task_ids) else 0.0
    
    print("\n========================================================")
    print(f"EVALUATION REPORT FOR {model_id} ({reasoning_effort.upper()}) [UNIFIED SENSE CLUSTERING]")
    print("========================================================")
    print(f"Clustering Precision:         {mean_precision:.2f}%")
    print(f"Clustering Recall:            {mean_recall:.2f}%")
    print(f"Clustering F1-Score:          {mean_f1:.2f}%")
    print(f"Bypassed by POS mismatch:     {resolved_by_pos_mismatch} senses")
    print(f"LLM Calls Made:               {llm_calls_made} / {len(test_task_ids)} tasks")
    print(f"LLM Bypassed Words Rate:      {llm_bypass_rate:.2f}%")
    print(f"Prompt (Input) Tokens:        {total_prompt_tokens}")
    print(f"Completion (Output) Tokens:    {total_completion_tokens}")
    print("========================================================")

    if errors_occurred:
        print("\n[WARNING] Some LLM requests failed during evaluation:")
        for w, err in errors_occurred:
            print(f"  - '{w}': {err}")

    # Step 4: Save evaluation history
    import datetime
    safe_model_name = model_id.replace("/", "_").replace("\\", "_")
    history_path = f"data/evaluations/{safe_model_name}_{reasoning_effort}.json"
    
    record = {
        "timestamp": datetime.datetime.now().astimezone().isoformat(),
        "model": model_id,
        "reasoning_effort": reasoning_effort,
        "limit": limit,
        "workers": workers,
        "metrics": {
            "clustering_precision": mean_precision,
            "clustering_recall": mean_recall,
            "clustering_f1": mean_f1,
            "resolved_by_pos_mismatch": resolved_by_pos_mismatch,
            "llm_calls_made": llm_calls_made,
            "total_tasks": len(test_task_ids),
            "llm_bypass_rate": llm_bypass_rate
        },
        "token_usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens
        },
        "detailed_samples": detailed_samples
    }
    
    history = []
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
                if not isinstance(history, list):
                    history = []
        except Exception as he:
            print(f"[Warning] Failed to load history from {history_path}: {he}")
            
    history.append(record)
    
    try:
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        print(f"\nEvaluation history successfully saved to: {history_path}")
    except Exception as se:
        print(f"[ERROR] Failed to save evaluation history: {se}")
