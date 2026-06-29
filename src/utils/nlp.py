import re

def map_pos(cam_pos):
    """Map Cambridge POS string to WordNet POS character."""
    if not cam_pos:
        return None
    pos = cam_pos.lower().strip()
    if 'noun' in pos:
        return 'n'
    if 'adverb' in pos or 'adv' in pos:
        return 'r'
    if 'verb' in pos:
        return 'v'
    if 'adjective' in pos or 'adj' in pos:
        return 'a'
    return None

def extract_int(val):
    """Robustly extract the first sequence of digits as an integer from a string."""
    digits = re.findall(r'\d+', val)
    if digits:
        return int(digits[0])
    raise ValueError(f"No digits found in: {val}")

import unicodedata

def resolve_alternatives(text: str):
    if '/' not in text:
        return [text]
    
    parts = text.split('/')
    if any(len(p.strip()) <= 1 for p in parts) or any(any(c.isdigit() for c in p) for p in parts):
        return [text]
        
    has_spaces = [(' ' in p.strip()) for p in parts]
    if all(has_spaces) or not any(has_spaces):
        return [p.strip() for p in parts if p.strip()]
        
    match = re.search(r'\b(\w+)/(\w+)\b', text)
    if match:
        w1, w2 = match.group(1), match.group(2)
        span = match.span()
        text_v1 = text[:span[0]] + w1 + text[span[1]:]
        text_v2 = text[:span[0]] + w2 + text[span[1]:]
        return list(set(resolve_alternatives(text_v1) + resolve_alternatives(text_v2)))
    return [p.strip() for p in parts if p.strip()]

def get_match_keys(text: str):
    if not text:
        return []

    text = text.lower().strip()
    text = re.sub(r'^(the|a|an)\s+', '', text)
    text = "".join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    text = text.replace(".", "")

    if re.match(r'^[a-z],\s*[a-z]$', text):
        text = text[0]

    placeholder_slashes = r'\b(someone|something|somebody|somewhere|sth|sb|oneself|yourself|himself|herself|themselves)/+(someone|something|somebody|somewhere|sth|sb|oneself|yourself|himself|herself|themselves)\b'
    text = re.sub(placeholder_slashes, ' ', text, flags=re.IGNORECASE)

    placeholders = r'\b(someone|something|somebody|somewhere|sth|sb|oneself|yourself|himself|herself|themselves|doing)\b'
    text = re.sub(placeholders, ' ', text, flags=re.IGNORECASE)
    
    possessives = r"\b(someone's|somebody's|one's|your|their|his|her|my|our)\b"
    text = re.sub(possessives, ' ', text, flags=re.IGNORECASE)

    texts_to_resolve = [text]
    while True:
        new_texts = []
        has_bracket = False
        for t in texts_to_resolve:
            match = re.search(r'\(([^)]+)\)', t)
            if match:
                has_bracket = True
                span = match.span()
                inside_content = match.group(1)
                
                t_keep = t[:span[0]] + " " + inside_content + " " + t[span[1]:]
                t_drop = t[:span[0]] + " " + t[span[1]:]
                
                new_texts.append(t_keep)
                new_texts.append(t_drop)
            else:
                new_texts.append(t)
        
        texts_to_resolve = list(set(new_texts))
        if not has_bracket:
            break

    final_phrases = []
    for t in texts_to_resolve:
        final_phrases.extend(resolve_alternatives(t))

    candidates = []
    for phrase in set(final_phrases):
        phrase_clean = phrase.replace("-", " ").replace("_", " ").replace(":", " ")
        phrase_clean = re.sub(r'[^a-z0-9\s]', '', phrase_clean)
        phrase_clean = re.sub(r'\s+', ' ', phrase_clean).strip()
        
        if phrase_clean:
            candidates.append(phrase_clean)
            if not any(c.isdigit() for c in phrase_clean):
                squashed = phrase_clean.replace(" ", "")
                if squashed != phrase_clean:
                    candidates.append(squashed)

    return list(set(candidates))
