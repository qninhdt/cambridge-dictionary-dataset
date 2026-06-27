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
