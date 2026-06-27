import re
import time
import requests
import json
from src.utils.nlp import extract_int


def build_alignment_prompt(batch_tasks):
    """
    Construct professional system instructions and a structured JSON-based user prompt 
    for aligning one or more words/tasks in an LLM request into Unified Senses (M-to-N).

    Parameters:
        batch_tasks: List of dicts, each containing:
            - 'task_id': unique identifier (str)
            - 'word': target word (str)
            - 'cam_senses': List of tuples (s_id, pos, definition)
            - 'wn_synsets': List of WordNet Synset objects or dicts
    """
    system_instruction = (
        "You are a professional lexicographer specializing in combining dictionary datasets. "
        "Your task is to merge senses from the Cambridge Advanced Learner's Dictionary (CALD) and WordNet synsets "
        "for a given word and POS into a set of 'Unified Senses' (distinct semantic concepts) "
        "to be used in an English learning application for language students.\n\n"
        "### Rules for Creating Unified Senses:\n"
        "1. **Semantic Merging (M-to-N)**: Group matching CALD definitions and WordNet synsets that describe "
        "the same core concept. Each group represents a 'Unified Sense'.\n"
        "2. **Single-Sentence Definition**: For each Unified Sense, write a new definition summarizing the concept "
        "in exactly ONE short, clear, simple sentence tailored for language learners (learner-friendly vocabulary, no overly complex or obscure wording).\n"
        "3. **Complete Coverage Constraint**:\n"
        "   - Every input Cambridge sense MUST appear in at least one unified sense.\n"
        "   - Every input WordNet synset MUST appear in at least one unified sense.\n"
        "   - If a sense/synset has no counterpart in the other dictionary, it MUST still form its own Unified Sense "
        "with the other dictionary's index list being empty [].\n"
        "4. **Split Senses**: If an input sense is coarse and spans multiple distinct concepts in the other dictionary, "
        "it can appear in multiple Unified Senses.\n\n"
        "### Response Format:\n"
        "You MUST respond with a single, valid JSON object matching this schema. Do not write any markdown code blocks, "
        "no wrapping in ```json, and no introductory or concluding text. Output only raw JSON:\n"
        "{\n"
        "  \"results\": [\n"
        "    {\n"
        "      \"task_id\": \"word_pos\",\n"
        "      \"unified_senses\": [\n"
        "        {\n"
        "          \"definition\": \"Exactly one short sentence definition.\",\n"
        "          \"cambridge_senses\": [1],\n"
        "          \"wordnet_synsets\": [1, 2]\n"
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "### Example Task & Expected JSON Response:\n\n"
        "### Example Task:\n"
        "<example_task>\n"
        "<alignment_tasks>\n"
        "  <task id=\"accessory_n\">\n"
        "    <target_word>accessory</target_word>\n"
        "    <cambridge_senses>\n"
        "      [1] something added to a machine or to clothing that has a useful or decorative purpose\n"
        "      [2] someone who helps another person to commit a crime but does not take part in it\n"
        "    </cambridge_senses>\n"
        "    <wordnet_synsets>\n"
        "      [1] clothing that is worn or carried, but not part of your main clothing\n"
        "      [2] a supplementary component that improves capability\n"
        "      [3] someone who helps another person commit a crime\n"
        "    </wordnet_synsets>\n"
        "  </task>\n"
        "</alignment_tasks>\n"
        "</example_task>\n\n"
        "### Expected JSON Response:\n"
        "{\n"
        "  \"results\": [\n"
        "    {\n"
        "      \"task_id\": \"accessory_n\",\n"
        "      \"unified_senses\": [\n"
        "        {\n"
        "          \"definition\": \"An item of clothing worn or carried to complement main clothing.\",\n"
        "          \"cambridge_senses\": [1],\n"
        "          \"wordnet_synsets\": [1]\n"
        "        },\n"
        "        {\n"
        "          \"definition\": \"A supplementary component that improves a machine's capability.\",\n"
        "          \"cambridge_senses\": [1],\n"
        "          \"wordnet_synsets\": [2]\n"
        "        },\n"
        "        {\n"
        "          \"definition\": \"Someone who helps another person commit a crime.\",\n"
        "          \"cambridge_senses\": [2],\n"
        "          \"wordnet_synsets\": [3]\n"
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}"
    )

    user_prompt = "<alignment_tasks>\n"
    batch_lookups = {}

    for task in batch_tasks:
        task_id = task["task_id"]
        word = task["word"]
        cam_senses = task["cam_senses"]
        wn_synsets = task["wn_synsets"]

        # Index Cambridge senses with integers (1, 2, ...)
        cam_lookup = {}
        cam_text = ""
        for idx, (s_id, pos, definition) in enumerate(cam_senses):
            short_id = idx + 1
            cam_lookup[short_id] = {"id": s_id, "definition": definition, "pos": pos}
            cam_text += f"[{short_id}] {definition}\n"

        # Index WordNet synsets with integers (1, 2, ...)
        wn_lookup = {}
        wn_text = ""
        for idx, ss in enumerate(wn_synsets):
            short_id = idx + 1
            if isinstance(ss, dict):
                ss_id = ss["id"]
                ss_def = ss["definition"]
                ss_pos = ss["pos"]
            else:
                ss_id = ss.id
                ss_def = ss.definition()
                ss_pos = ss.pos
            wn_lookup[short_id] = {"id": ss_id, "definition": ss_def, "pos": ss_pos}
            wn_text += f"[{short_id}] {ss_def}\n"

        batch_lookups[task_id] = (cam_lookup, wn_lookup)

        user_prompt += f"""  <task id="{task_id}">
    <target_word>{word}</target_word>
    <cambridge_senses>
{cam_text.strip()}
    </cambridge_senses>
    <wordnet_synsets>
{wn_text.strip()}
    </wordnet_synsets>
  </task>\n\n"""

    user_prompt += "</alignment_tasks>\n\nRESPONSE CONSTRAINT: You MUST respond with a single valid raw JSON object. Do not wrap in markdown tags or include conversational text."
    return system_instruction, user_prompt, batch_lookups


def parse_alignment_response(content, batch_lookups):
    """
    Parse JSON alignment response and map indices back to database/WordNet IDs.

    Returns:
        dict: {
            task_id: [
                {
                    "definition": str,
                    "cambridge_sense_ids": list,
                    "wordnet_synset_ids": list
                },
                ...
            ]
        }
    """
    batched_results = {}

    # Strip thinking blocks
    cleaned_content = re.sub(r"(?s)<think>.*?</think>", "", content).strip()
    # Strip markdown formatting
    cleaned_content = re.sub(r"```json\s*|\s*```", "", cleaned_content).strip()

    try:
        data = json.loads(cleaned_content)
        if isinstance(data, dict) and "results" in data:
            results_list = data["results"]
        elif isinstance(data, list):
            results_list = data
        else:
            results_list = [data]
    except Exception as e:
        print(f"Error parsing JSON response: {e}\nContent was:\n{cleaned_content}")
        return None

    for task_item in results_list:
        if not isinstance(task_item, dict):
            continue
        task_id = task_item.get("task_id")
        if task_id not in batch_lookups:
            continue

        cam_lookup, wn_lookup = batch_lookups[task_id]
        unified_list = task_item.get("unified_senses", [])
        if not unified_list:
            continue

        task_senses = []
        for sense_item in unified_list:
            if not isinstance(sense_item, dict):
                continue
            definition = sense_item.get("definition", "").strip()
            cam_indices = sense_item.get("cambridge_senses", [])
            wn_indices = sense_item.get("wordnet_synsets", [])

            # Map Cambridge indices to DB IDs
            cam_ids = []
            for idx in cam_indices:
                try:
                    info = cam_lookup.get(int(idx))
                    if info:
                        cam_ids.append(int(info["id"]))
                except (ValueError, TypeError):
                    continue

            # Map WordNet indices to Synset IDs
            wn_ids = []
            for idx in wn_indices:
                try:
                    info = wn_lookup.get(int(idx))
                    if info:
                        wn_ids.append(info["id"])
                except (ValueError, TypeError):
                    continue

            task_senses.append({
                "definition": definition,
                "cambridge_sense_ids": cam_ids,
                "wordnet_synset_ids": wn_ids
            })
            
        if task_senses:
            batched_results[task_id] = task_senses

    return batched_results


def call_alignment_llm(
    model_name, system_prompt, user_prompt, api_key, api_base, effort
):
    """Send alignment prompt to the OpenAI-compatible API and retrieve token details."""
    import hashlib
    import json
    import os

    safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    cache_dir = os.path.join("data/llm_cache", safe_model_name, effort or "none")

    # Compute cache key from prompts and effort
    request_data = {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "effort": effort,
    }
    request_str = json.dumps(request_data, sort_keys=True)
    cache_key = hashlib.sha256(request_str.encode("utf-8")).hexdigest()
    cache_path = os.path.join(cache_dir, f"{cache_key}.json")

    # Load from cache if exact match exists
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            cached_res = cached_data.get("response", {})
            if "content" in cached_res:
                print(
                    f"  [Cache Hit] Using cached response for {model_name} (Hash: {cache_key[:8]})"
                )
                return cached_res
        except Exception as ce:
            print(f"  [Warning] Failed to read LLM cache at {cache_path}: {ce}")

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=api_base)

    create_params = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_completion_tokens": 8000
    }
    
    # Set temperature to 0.0 for deterministic alignment outputs.
    # Exclude older reasoning models (o1-mini, o1-preview) which do not support the temperature parameter.
    if not (model_name.startswith("o1-mini") or model_name.startswith("o1-preview")):
        create_params["temperature"] = 0.0
        
    if effort and effort.lower() != "none":
        create_params["reasoning_effort"] = effort.lower()

    attempt = 0

    while True:
        attempt += 1
        try:
            response = client.chat.completions.create(**create_params)
            content = response.choices[0].message.content or ""

            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            
            reasoning_tokens = 0
            if usage and hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
                reasoning_tokens = getattr(usage.completion_tokens_details, "reasoning_tokens", 0)

            res_payload = {
                "content": content.strip(),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": reasoning_tokens,
                "error": None,
            }
            cache_payload = {
                "input": {
                    "model": model_name,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "effort": effort,
                },
                "response": res_payload,
            }
            try:
                os.makedirs(cache_dir, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache_payload, f, indent=2)
            except Exception as cs:
                print(f"  [Warning] Failed to write LLM cache: {cs}")
            return res_payload
        except Exception as e:
            print(f"  [Warning] Attempt {attempt} error: {e}. Retrying in 5 seconds...")
            time.sleep(5)
