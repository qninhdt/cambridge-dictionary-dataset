import re

VERB_PREFIXES = {
    'trickle', 'think', 'have', 'set', 'declare', 'go', 'move', 'return',
    'do', 'does', 'did', 'make', 'take', 'bring', 'keep', 'put', 'play',
    'give', 'get', 'run', 'send', 'throw', 'tell', 'write', 'be', 'is',
    'are', 'was', 'were', 'been', 'redeem', 'strike',
    'spirit', 'wave', 'catch', 'lighten', 'pique', 'repay', 'tread', 'come', 'pour'
}

def adjust_articles(choices):
    has_article_prefix = any(c.lower().startswith('a ') or c.lower().startswith('an ') for c in choices)
    if has_article_prefix:
        new_choices = []
        for c in choices:
            c_clean = c.strip()
            if c_clean.lower().startswith('a '):
                noun = c_clean[2:]
            elif c_clean.lower().startswith('an '):
                noun = c_clean[3:]
            else:
                noun = c_clean
                
            if noun and c_clean == noun:
                if noun.endswith('s') or noun.endswith("'") or noun.endswith("s'"):
                    new_choices.append(noun)
                else:
                    first_letter = re.sub(r'[^a-zA-Z]', '', noun)[:1].lower()
                    if first_letter in {'a', 'e', 'i', 'o', 'u'}:
                        new_choices.append(f"an {noun}")
                    else:
                        new_choices.append(f"a {noun}")
            else:
                if noun:
                    first_letter = re.sub(r'[^a-zA-Z]', '', noun)[:1].lower()
                    if first_letter in {'a', 'e', 'i', 'o', 'u'}:
                        new_choices.append(f"an {noun}")
                    else:
                        new_choices.append(f"a {noun}")
                else:
                    new_choices.append(c_clean)
        return new_choices
    return choices

def factor_choices(choices):
    if not choices or len(choices) <= 1:
        return "", choices, ""
    
    tokenized = [c.split() for c in choices]
    common_suffix = []
    min_len = min(len(t) for t in tokenized)
    for i in range(1, min_len + 1):
        last_token = tokenized[0][-i]
        last_token_clean = re.sub(r'[{}[\]()]', '', last_token).lower()
        if all(
            re.sub(r'[{}[\]()]', '', tokenized[j][-i]).lower().startswith(last_token_clean) or
            last_token_clean.startswith(re.sub(r'[{}[\]()]', '', tokenized[j][-i]).lower())
            for j in range(len(tokenized))
        ):
            longest_token = max((tokenized[j][-i] for j in range(len(tokenized))), key=len)
            common_suffix.insert(0, longest_token)
        else:
            break
            
    if common_suffix:
        for j in range(len(tokenized)):
            tokenized[j] = tokenized[j][:-len(common_suffix)]
            
    common_prefix = []
    min_len = min(len(t) for t in tokenized)
    for i in range(min_len):
        first_token = tokenized[0][i]
        if all(tokenized[j][i] == first_token for j in range(len(tokenized))):
            common_prefix.append(first_token)
        else:
            break
            
    if common_prefix:
        for j in range(len(tokenized)):
            tokenized[j] = tokenized[j][len(common_prefix):]
            
    choices_rem = [" ".join(t) for t in tokenized]
    
    # Extra-prefix factoring: pull out leading words unique to choices_rem[0] when those words
    # are function words / known verb prefixes AND all other choices are shorter.
    # Guard: skip factoring if the candidate prefix word appears as the FIRST word of any other
    # choice — that means it's a genuine alternative, not a shared prefix.
    # Example allowed:   ["catch {possessive} attention", "imagination"] -> "catch {possessive}" + ["attention", "imagination"]
    # Example prevented: ["do {sth}", "nothing", "{any}"] -> NO factoring (do != nothing/any)
    FACTORABLE_WORDS = VERB_PREFIXES | {
        'at', 'in', 'on', 'with', 'for', 'by', 'under', 'through', 'from', 'into',
        'a', 'an', 'the', 'its', 'your', 'his', 'her', 'their', 'our', 'my',
        'space', 'of', 'somewhere', 'something', 'someone', 'somebody',
        'sw', 'sth', 'sb', 'possessive', 'end', 'side', 'times',
        'spirit', 'wave', 'catch', 'lighten', 'pique', 'repay', 'tread',
        'come', 'go', 'grow', 'pour', 'throw', 'like',
    }
    extra_prefix = ""
    if len(choices_rem) > 1 and len(choices_rem[0].split()) > 1:
        c0_words = choices_rem[0].split()
        max_other_len = max(len(c.split()) for c in choices_rem[1:])
        num_to_factor = len(c0_words) - max_other_len
        if num_to_factor > 0:
            # Compute how many leading words of c0 are factorable function-words
            factorable_len = 0
            for i in range(num_to_factor):
                w_clean = re.sub(r'[{}[\]()]', '', c0_words[i]).lower()
                if '/' in w_clean or w_clean in FACTORABLE_WORDS:
                    factorable_len += 1
                else:
                    break
            # Guard: if a candidate prefix word appears as the leading word of ANY other choice,
            # those choices are genuine alternatives — do NOT factor that word out.
            if factorable_len > 0:
                first_factor_clean = re.sub(r'[{}[\]()]', '', c0_words[0]).lower()
                if any(
                    other.split() and re.sub(r'[{}[\]()]', '', other.split()[0]).lower() == first_factor_clean
                    for other in choices_rem[1:]
                ):
                    factorable_len = 0  # first candidate prefix is used as alt — don't factor
            if factorable_len > 0:
                factored = c0_words[:factorable_len]
                choices_rem[0] = " ".join(c0_words[factorable_len:])
                extra_prefix = " ".join(factored) + " "
                    
    prefix_str = " ".join(common_prefix)
    if extra_prefix:
        prefix_str = (prefix_str + " " + extra_prefix).strip()
    suffix_str = " ".join(common_suffix)
    
    return prefix_str, choices_rem, suffix_str

def flatten_pipes(text):
    while True:
        new_text = re.sub(r'\(\(([^()]+)\)\|', r'(\1|', text)
        new_text = re.sub(r'\|\(([^()]+)\)\)', r'|\1)', new_text)
        new_text = re.sub(r'\|\(([^()]+)\)\|', r'|\1|', new_text)
        if new_text == text:
            break
        text = new_text
    return text


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
    text = text.replace("’", "'")
    # Đổi dấu gạch dưới, gạch ngang, hai chấm thành khoảng trắng ngay từ đầu để tránh lỗi regex
    text = text.replace("_", " ").replace("-", " ").replace(":", " ")
    
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
        phrase_clean = re.sub(r'[^a-z0-9\s]', '', phrase)
        phrase_clean = re.sub(r'\s+', ' ', phrase_clean).strip()
        
        if phrase_clean:
            # Chặn nếu key sau khi rút gọn chỉ còn lại 1 từ thuộc STOP_WORDS để tránh khớp sai
            if phrase_clean in STOP_WORDS:
                continue
                
            candidates.append(phrase_clean)
            if not any(c.isdigit() for c in phrase_clean):
                squashed = phrase_clean.replace(" ", "")
                if squashed != phrase_clean and squashed not in STOP_WORDS:
                    candidates.append(squashed)

    return list(set(candidates))

STOP_WORDS = {
    'a', 'an', 'the', 'to', 'of', 'in', 'at', 'on', 'with', 'for', 'and', 'or', 
    'as', 'so', 'if', 'when', 'by', 'someone', 'something', 'somebody', 'sb', 'sth', 'sw', 'etc', 's'
}

def normalize_phrase(text: str, is_single_word: bool = False) -> str:
    # Step 1: Preprocessing & HTML Unescaping
    text = text.replace("&amp;", "&")
    text = text.replace("’", "'")
    text = re.sub(r'\s*/\s*', '/', text)
    text = text.strip(" /")
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    
    # Step 2: Unicode Accent Flattening
    text = unicodedata.normalize('NFD', text)
    text = "".join(c for c in text if unicodedata.category(c) != 'Mn')
    
    # Step 3: Special Character Replacements
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("×", "x")
    
    # Step 7 (Moved early): Translate Optionals & Suffixes (Parentheses to Brackets)
    text = re.sub(r'(\w+)\(([^0-9)]+)\)', r'\1[\2]', text)
    text = re.sub(r'\(([^0-9)]+)\)', r'[\1]', text)

    # Step 4a: Syntactic Phrase-Level Slash Expansion & Formatting Corrections
    # Pre-process /etc. at the end of a slash list to , etc.
    text = re.sub(r'/etc\.?$', ', etc.', text)
    text = re.sub(r'/etc\.?\b', ', etc.', text)

    # Hyphen-prefix lists: "one-/two-/three-, etc. WORD" -> "(one-WORD|two-WORD|three-WORD|{any})"
    # Must run before other replacements so the suffix word is captured.
    def _expand_hyphen_list(m):
        prefix_list_str = m.group(1)  # e.g. "one-/two-/three-/four-"
        suffix = m.group(2)           # e.g. "alarm"
        # Split by slash, strip trailing hyphens to get stems, then rejoin with suffix
        stems = [p.rstrip('-') for p in prefix_list_str.split('/') if p.strip('-')]
        options = [f"{s}-{suffix}" for s in stems]
        options.append("{any}")
        return f"({'|'.join(options)})"
    # Pattern: sequence of "word-/word-/..." followed by optional ", etc." and then a word
    text = re.sub(
        r'((?:[\w]+-/)+[\w]+-),?\s*etc\.\s+([\w]+)',
        _expand_hyphen_list,
        text
    )

    # General / specific phrase-level slash grouping & distributions
    text = text.replace("go into/move into/return to, etc.", "go into/move into/return to, etc.")
    text = text.replace("do you/did you/does he, etc.", "do you/did you/does he, etc.")
    text = text.replace("easy as pie/abc/anything/falling off a log", "easy as (pie|abc|anything|falling off a log)")
    text = text.replace("mad as a hatter/march hare", "mad as (a hatter|march hare)")
    text = text.replace("a fortune/a bomb/the earth", "(a fortune|a bomb|the earth)")
    text = text.replace("curve [ball]/curveball", "(curve [ball]|curveball)")
    text = text.replace("in/out of keeping", "(in|out of) keeping")
    text = text.replace("in/out of tune", "(in|out of) tune")
    text = text.replace("against/in favour of", "(against|in favour of)")
    text = text.replace("strike a deal/agreement", "strike (a deal|an agreement)")
    text = text.replace("be hit hard/be hard hit", "(be hit hard|be hard hit)")
    # be knocking (on) + number list: keep "be knocking [on]" as prefix
    text = re.sub(
        r'\bbe\s+knocking\s+(\[on\]|\(on\))\s+([\d,\s]+(?:,\s*etc\.)?)',
        lambda m: f'be knocking [{"on"}] (' + '|'.join(p.strip() for p in re.split(r',', m.group(2).replace(', etc.', '').replace('etc.', '').strip()) if p.strip()) + f'|{{any}})',
        text
    )
    text = text.replace("declare an interest/a conflict of interest(s)", "declare (an interest|a conflict of interest(s))")
    text = text.replace("declare an interest/a conflict of interest", "declare (an interest|a conflict of interest)")
    text = text.replace("be going to do/be", "be going to (do|be)")
    # Set store by: "set a lot of, great, little, etc. store by" -> "set (a lot of|great|little|{any}) store by"
    text = re.sub(
        r'\bset\s+a\s+lot\s+of,\s+great,\s+little,\s+etc\.\s+store\s+by\b',
        'set (a lot of|great|little|{any}) store by',
        text
    )
    # Be between + list: "be between X/Y/Z, W, etc." -> "be between (X|Y|Z|W|{any})"
    text = re.sub(
        r'\bbe\s+between\s+([\w/]+(?:,\s*[\w]+)*),\s*etc\.',
        lambda m: 'be between (' + '|'.join(p.strip() for p in re.split(r'[/,]+', m.group(1)) if p.strip()) + '|{any})',
        text
    )
    # someone's path recrosses/paths recross — both subjects differ
    text = re.sub(
        r"someone's\s+path\s+recrosses/paths\s+recross",
        "(someone's path recrosses|paths recross)",
        text
    )
    # "much to {possessive}/someone's WORD, WORD, etc." -> "much to {possessive} (WORD|WORD|{any})"
    # Must run after Step 5 placeholder unification (which converts someone's -> {possessive})
    # so we match {possessive} here, but this Step 4a runs before Step 5.
    # Handle at pre-Step-5 level with the raw word:
    text = re.sub(
        r"\bmuch\s+to\s+(?:someone's|somebody's|{possessive})\s+([\w\s,]+),\s*etc\.",
        lambda m: 'much to {possessive} (' + '|'.join(p.strip() for p in m.group(1).split(',') if p.strip()) + '|{any})',
        text
    )
    
    # Generic double / triple slash with doing / sth
    text = re.sub(r'\bfor/to do (something|sth)\b', r'(for something|to do something)', text)
    text = re.sub(r'\bfor (something|sth)/to do (something|sth)\b', r'(for something|to do something)', text)
    text = re.sub(r'\bto (something|sth)/do (something|sth)\b', r'to (something|do something)', text)
    text = re.sub(r'\ba billion (something|sth)/billions of (something|sth)\b', r'(a billion something|billions of something)', text)
    
    # beyond/out of (sb's) reach — 'out of' atomic + use {possessive}
    # Note: Step 7 runs early and converts (sb's) -> [sb's], so match both forms
    text = re.sub(
        r"\bbeyond/out\s+of\s+(?:\(?(?:sb's?|someone's?|somebody's?)\)?|\[(?:sb's?|someone's?|somebody's?)\])\s+reach\b",
        '(beyond|out of) [{possessive}] reach',
        text
    )
    # Generic word-level split of bargepole/barge pole
    text = re.sub(r'\b([a-zA-Z]+)([a-zA-Z]+)/\1\s+\2\b', r'(\1\2|\1 \2)', text)

    # Step 4b: Protect Literal Collocations of Placeholders from Unification (Sorted by length descending)
    protection = [
        # Protect 'something like that' (fixed idiom, not a variable)
        (r'\b(something)\s+like\s+that\b', r'__LIT_\1__ like that'),
        # Protect 'be my/your/his/her/their doing' where 'doing' is a noun (not V-ing placeholder)
        # Only for personal pronouns, NOT for 'someone\'s doing' which remains {doing}
        (r'\bbe\s+(my|your|his|her|their|our|its)\s+(doing)\b', r'be \1 __LIT_\2__'),
        (r'\bno\s+sooner\s+does\s+(something)\s+happen\s+than\s+(something)\s+else\s+happens\b', r'no sooner does __LIT_\1__ happen than __LIT_\2__ else happens'),
        (r'\b(somewhere)\s+along\s+the\s+line\b', r'__LIT_\1__ along the line'),
        (r'\b(somewhere)\s+around,\s+between,\s+etc\.', r'__LIT_\1__ around, between, etc.'),
        (r'\b(somewhere)\s+in\s+the\s+region\s+of\b', r'__LIT_\1__ in the region of'),
        (r'\b(somewhere)\s+down\s+the\s+road\b', r'__LIT_\1__ down the road'),
        (r'\bget\s+far/(somewhere)/anywhere\b', r'get far/__LIT_\1__/anywhere'),
        (r"\bthere's\s+(something)\s+to\s+be\s+said\s+for\b", r"there's __LIT_\1__ to be said for"),
        (r"\bthere's\s+(something)\s+about\b", r"there's __LIT_\1__ about"),
        (r'\bbe\s+getting\s+(somewhere)\b', r'be getting __LIT_\1__'),
        (r'\bbe\s+(something)\s+of\s+a\b', r'be __LIT_\1__ of a'),
        # Protect 'something like N...' when used as adverb meaning 'approximately'
        (r'\b(something)\s+like\s+(?=\d)', r'__LIT_\1__ like '),
        (r'\b(somewhere)\s+(around|between|near|in\s+the\s+middle)\b', r'__LIT_\1__ \2'),
        (r'\b(someone|something)\s+or\s+other\b', r'__LIT_\1__ or other'),
        (r'\b(someone|something)\s+or\s+another\b', r'__LIT_\1__ or another'),
        (r'\ba\s+little\s+(something)\b', r'a little __LIT_\1__'),
        (r'\b(something)\s+for\s+nothing\b', r'__LIT_\1__ for nothing'),
        (r'\b(something)\s+fierce\b', r'__LIT_\1__ fierce'),
        (r'\b(something)\s+(along\s+those\s+lines|to\s+that\s+effect|to\s+write\s+home\s+about|in\s+the\s+air|big)\b', r'__LIT_\1__ \2'),
        (r'\b(something)\s+on\s+(your|his|her|my|our|their|its|{possessive})\s+mind\b', r'__LIT_\1__ on \2 mind'),
        (r"\b(something)'s\s+(got\s+to\s+give|cooking)\b", r"__LIT_\1__'s \2"),
        (r'(?<!/)\b(someone|something|somewhere)\s+else\b', r'__LIT_\1__ else'),
        (r'\b(get|getting)\s+(somewhere)\b', r'\1 __LIT_\2__'),
        (r'\bor\s+(something|somewhere)\b', r'or __LIT_\1__'),
        (r'\byou\s+know\s+(something)\b', r'you know __LIT_\1__'),
        (r'\b(take\s+some|nothing|what\s+you\s+are|can\'t\s+be)\s+(doing)\b', r'\1 __LIT_\2__'),
        (r'\bhave\s+got\s+(something)\s+there\b', r'have got __LIT_\1__ there'),
        (r'\b(something)\s+tells\s+me\b', r'__LIT_\1__ tells me'),
        (r'\bmake\s+(something)\s+of\b', r'make __LIT_\1__ of'),
        (r'\b(really|quite)\s+(something|somebody|someone)\b', r'\1 __LIT_\2__'),
        (r'\b([a-zA-Z]+)-(something)\b', r'\1-__LIT_\2__'),
        (r'\bis\s+that\s+(something)\b', r'is that __LIT_\1__'),
        (r'\b(be|become)\s+(somebody|someone)(?!\'s)\b', r'\1 __LIT_\2__'),
        (r'\b(someone)\s+of\s+importance\b', r'__LIT_\1__ of importance')
    ]
    for pattern, protected in protection:
        text = re.sub(pattern, protected, text)
        
    # Step 5: Compound Pronouns & Placeholder Unification (Possessives run first to prevent someone's -> {sb}'s)
    if not is_single_word:
        placeholders = [
            (r'\b(?:his/her|his or her|one\'s|ones\'|someone\'s|somebody\'s|your|their|his|her|my|our|its)\b', "{possessive}"),
            (r'\b(?:him/her|he/she|he or she|him or her|someone|somebody|sb)\b', "{sb}"),
            (r'\b(?:something|sth)\b', "{sth}"),
            (r'\b(?:somewhere|someplace|sw)\b', "{sw}"),
            (r'\b(?:oneself|yourself|himself|herself|itself|themselves|myself|ourselves|yourselves)\b', "{oneself}"),
            (r'\bdoing\b', "{doing}")
        ]
        for pattern, rep in placeholders:
            text = re.sub(pattern, rep, text)
        # Collapse duplicate unified placeholder slashes (e.g. {possessive}/{possessive} -> {possessive})
        text = re.sub(r'\{possessive\}(?:/\{possessive\})+', '{possessive}', text)
        text = re.sub(r'\{sb\}(?:/\{sb\})+', '{sb}', text)
        text = re.sub(r'\{sth\}(?:/\{sth\})+', '{sth}', text)
        text = re.sub(r'\{sw\}(?:/\{sw\})+', '{sw}', text)
        text = re.sub(r'\{oneself\}(?:/\{oneself\})+', '{oneself}', text)
        # ({possessive}|{any}) -> {possessive}: when the only named alternatives are possessives,
        # {any} is entirely redundant (it already covers possessives).
        text = re.sub(r'\(\{possessive\}\|\{any\}\)', '{possessive}', text)
        
    # Step 5b: Restore Literal Placeholder Words
    text = text.replace("__LIT_somewhere__", "somewhere")
    text = text.replace("__LIT_something__", "something")
    text = text.replace("__LIT_someone__", "someone")
    text = text.replace("__LIT_somebody__", "somebody")
    text = text.replace("__LIT_doing__", "doing")
    
    # Post-Step5 hardcodes (run after placeholder unification so {sb}/{sth} won't be re-matched)
    # "jolt {sb} into/out of {sth}" — 'out of' is atomic
    text = re.sub(r'\bjolt\s+\{sb\}\s+into/out\s+of\s+\{sth\}\b',
                  lambda _: 'jolt {sb} (into|out of) {sth}', text)
    # Also handle the pre-placeholder form in case normalize_phrase is called directly
    text = re.sub(r'\bjolt\s+(?:someone|somebody)\s+into/out\s+of\s+(?:something)\b',
                  lambda _: 'jolt {sb} (into|out of) {sth}', text)
    # "{sb} couldn't ACT, VERB, VERB, etc." — after placeholder unification
    text = re.sub(
        r"\{sb\}\s+(couldn't|can't|won't|wouldn't)\s+(\w+),\s*((?:\w+,\s*)+)etc\.",
        lambda m: '{sb} ' + m.group(1) + ' (' + '|'.join(
            [m.group(2)] + [p.strip() for p in m.group(3).rstrip(', ').split(',') if p.strip()]
        ) + '|{any})',
        text
    )
    
    
    # Step 7b: Translate Slashes inside Brackets (Optionals Choices)
    def replace_bracket_with_slash(match):
        content = match.group(1)
        
        # Check if we should split this bracket
        has_space = ' ' in content
        slash_count = content.count('/')
        ends_with_etc = content.strip(']. ').endswith('etc')
        
        should_split = not has_space or slash_count == 1 or ends_with_etc
        if not should_split:
            return match.group(0)
            
        parts = [p.strip() for p in content.split('/') if p.strip()]
        normalized_parts = []
        for p in parts:
            if p.strip('.') == 'etc':
                normalized_parts.append("{any}")
            else:
                normalized_parts.append(p)
        parts = normalized_parts
        
        # Prefix Factoring
        if len(parts) > 1:
            first_words = parts[0].split()
            if len(first_words) > 1:
                all_others_single = all(len(p.split()) == 1 for p in parts[1:])
                if all_others_single:
                    prefix = " ".join(first_words[:-1])
                    parts[0] = first_words[-1]
                    return f"[{prefix} ({'|'.join(parts)})]"
                    
        return f"[{'|'.join(parts)}]"
    text = re.sub(r'\[([^\]]*/[^\]]*)\]', replace_bracket_with_slash, text)
    
    # Step 8: Slot Insertion & List Unification (word1, word2, etc. notice) with Prefix/Suffix Factoring
    while True:
        list_match = re.search(r'(?:^|(?<=\s))([\w\s\'/{}[\],\$£€¥§\-/]+?)(?:,\s*|\s+)etc\.', text)
        if not list_match:
            break
            
        list_str = list_match.group(1)
        choices = [p.strip() for p in re.split(r',(?!\d)', list_str) if p.strip()]
        prefix_words = ""
        if len(choices) == 1:
            words = list_str.split()
            if len(words) > 1:
                if '/' in words[-1]:
                    prefix_words = " ".join(words[:-1]) + " "
                    list_str = words[-1]
                    choices = [p.strip() for p in list_str.split('/') if p.strip()]
                elif words[-1].startswith('{') and words[-1].endswith('}'):
                    prefix_words = " ".join(words[:-1]) + " "
                    list_str = words[-1]
                    choices = [list_str]
                else:
                    choices = [p.strip() for p in list_str.split('/') if p.strip()]
            else:
                choices = [p.strip() for p in list_str.split('/') if p.strip()]
        
        # Remove duplicates preserving order
        unique_choices = []
        for c in choices:
            if c not in unique_choices:
                unique_choices.append(c)
        choices = unique_choices
        
        # Factor out prefixes / suffixes
        prefix_extracted, choices, suffix_extracted = factor_choices(choices)
        
        choices.append("{any}")
        
        # Remove duplicates preserving order
        unique_choices = []
        for c in choices:
            if c not in unique_choices:
                unique_choices.append(c)
        choices = unique_choices
        
        choice_group = f"({ '|'.join(choices) })"
        
        parts_rep = []
        if prefix_words:
            parts_rep.append(prefix_words.strip())
        if prefix_extracted:
            parts_rep.append(prefix_extracted)
        parts_rep.append(choice_group)
        if suffix_extracted:
            parts_rep.append(suffix_extracted)
            
        replacement = " ".join(parts_rep)
        text = text[:list_match.start()] + replacement + text[list_match.end():]
        
    # Step 9: Semicolon Alternative Splitting (Moved after Optionals to prevent bracket leakage)
    if ';' in text:
        parts = [p.strip() for p in text.split(';')]
        text = f"({'|'.join(parts)})"
        
    # Step 10: Compound Placeholders Unification (run after simple placeholders and optionals)
    if not is_single_word:
        compounds = [
            (r'\{sb\}/\{sth\}/\{doing\}\s+\{sth\}', "({sb}|{sth}|{doing} {sth})"),
            (r'\{sth\}/\{sb\}/\{doing\}\s+\{sth\}', "({sth}|{sb}|{doing} {sth})"),
            (r'\{sth\}/\{doing\}\s+\{sth\}', "({sth}|{doing} {sth})"),
            (r'\{sb\}/\{doing\}\s+\{sth\}', "({sb}|{doing} {sth})"),
            (r'\{doing\}\s+\{sth\}/\{sth\}', "({doing} {sth}|{sth})")
        ]
        for pattern, rep in compounds:
            text = re.sub(pattern, rep, text)
            
    # Step 10b: Pre-merge slash with following article/possessive if it makes a phrase choice
    def merge_slash_article(match):
        w1, w2, w3 = match.group(1), match.group(2), match.group(3)
        articles = {'a', 'an', 'the', 'your', 'his', 'her', 'my', 'our', 'their', 'its'}
        if w1.lower() in articles:
            return match.group(0)
        return f"({w1}|{w2} {w3})"
    text = re.sub(r'\b([a-zA-Z]+)/(a|an|the|your|his|her|my|our|their)\s+([a-zA-Z]+)\b', merge_slash_article, text)
    
    # Step 11: Translate Word-Level Alternatives (Slashes to Pipes)
    def replace_slash(match):
        segment = match.group(0)
        if re.search(r'\d/\d|\ba/c\b|\ba/s/l\b', segment):
            return segment
        parts = segment.split('/')
        
        # Remove duplicate parts preserving order
        unique_parts = []
        for p in parts:
            if p not in unique_parts:
                unique_parts.append(p)
        if len(unique_parts) == 1:
            return unique_parts[0]
        return f"({'|'.join(unique_parts)})"
        
    # Structured word pattern to prevent capturing surrounding braces/brackets
    word_pattern = r'(?:\{[a-zA-Z]+\}(?:\'[a-zA-Z]+)?|\[[a-zA-Z]+\]|[a-zA-Z]+(?:\'[a-zA-Z]+)+|[a-zA-Z]+(?:\[[a-zA-Z]+\])?|\'[a-zA-Z]+|[a-zA-Z]+\')'
    slash_pattern = word_pattern + r'(?:/' + word_pattern + r')+'
    text = re.sub(slash_pattern, replace_slash, text)
    
    # Step 12: Ellipses Mapping
    text = text.replace("...", "{any}")
    
    # Step 13: Punctuation Clean
    text = re.sub(r'[!?]+$', '', text)
    text = re.sub(r'\.(?!\S)', '', text)        # dot at end or before space
    text = re.sub(r'\.(?=[\])])', '', text)     # dot immediately before ] or )
    text = re.sub(r'\betc\b\.?', '', text)      # leftover 'etc' fragments not consumed by Step 8
    # Post-processing semantic fixups (run last, after all slash and bracket processing):
    # ({possessive}|{any}) is redundant — {any} already covers {possessive}
    text = re.sub(r'\(\{possessive\}\|\{any\}\)', '{possessive}', text)
    # "jolt {sb} (into|out) of {sth}" → "jolt {sb} (into|out of) {sth}" — 'out of' is atomic
    text = re.sub(r'\bjolt\s+\{sb\}\s+\(into\|out\)\s+of\s+\{sth\}\b',
                  lambda _: 'jolt {sb} (into|out of) {sth}', text)
    # declare an interest/a conflict of interest[s] — run after Step7 converts (s) -> [s]
    text = text.replace("(declare an interest|a conflict of interest[s])",
                        "declare (an interest|a conflict of interest[s])")
    text = text.replace("(declare an interest|a conflict of interest)",
                        "declare (an interest|a conflict of interest)")
    # be hit hard/be hard hit [by sth] — optional must be outside the alternation group
    text = text.replace("(be hit hard|be hard hit [by {sth}])",
                        "(be hit hard|be hard hit) [by {sth}]")
    text = re.sub(r'\s+', ' ', text).strip()
    
    return flatten_pipes(text)

def _hoist_common_suffix(normalized_parts):
    """Extract the longest common trailing word-sequence from all parts and hoist it outside.
    
    Treats (...) groups as single atomic tokens when scanning for shared suffix.
    """
    if not normalized_parts or len(normalized_parts) < 2:
        return normalized_parts, ""
    
    def tokenize(s):
        """Split into tokens, keeping (...) and [...] and {...} as single tokens."""
        tokens = []
        i = 0
        while i < len(s):
            if s[i] in '([{':
                close = {'{': '}', '[': ']', '(': ')'}[s[i]]
                depth = 1
                j = i + 1
                while j < len(s) and depth > 0:
                    if s[j] == s[i]:
                        depth += 1
                    elif s[j] == close:
                        depth -= 1
                    j += 1
                tokens.append(s[i:j])
                i = j
            elif s[i] == ' ':
                i += 1
            else:
                j = i
                while j < len(s) and s[j] not in ' ([{':
                    j += 1
                tokens.append(s[i:j])
                i = j
        return tokens
    
    tokenized = [tokenize(p) for p in normalized_parts]
    
    # find common suffix length
    min_len = min(len(t) for t in tokenized)
    common_len = 0
    for i in range(1, min_len + 1):
        if all(tokenized[j][-i] == tokenized[0][-i] for j in range(len(tokenized))):
            common_len = i
        else:
            break
    
    if common_len == 0:
        return normalized_parts, ""
    
    suffix = " ".join(tokenized[0][-common_len:])
    # For trimmed parts: if a part is a single (...) group, unwrap it to its inner alternatives
    def trim_and_unwrap(toks):
        toks = toks[:-common_len]
        joined = " ".join(toks)
        # Unwrap a bare (A|B|...) group so it can be flattened into the outer group
        if len(toks) == 1 and toks[0].startswith('(') and toks[0].endswith(')'):
            return toks[0][1:-1]  # return inner content, will be split by | in parent join
        return joined
    
    trimmed_inner = [trim_and_unwrap(t) for t in tokenized]
    return trimmed_inner, suffix

def normalize_expression(text: str) -> str:
    text = re.sub(r'\s*/\s*', '/', text)
    is_single_word = len(text.split()) == 1
    
    if '/' in text:
        if not re.search(r'\d/\d|\ba/c\b|\ba/s/l\b', text):
            parts = text.split('/')
            # If any slash-separated part contains 'etc.', it's a list expansion, not a phrase
            # alternative — skip phrase-level split and let normalize_phrase/Step8 handle it.
            if any('etc.' in p for p in parts):
                return normalize_phrase(text, is_single_word=is_single_word)

            is_phrase_level = False
            # Exclude trivial pronouns/articles from the shared-word heuristic to prevent
            # false positives like "if it/a thing is worth doing" triggering phrase-level split
            HEURISTIC_STOP = STOP_WORDS | {'it', 'a', 'an', 'the', 'if', 'that', 'this'}
            for i in range(len(parts) - 1):
                words_left = set(w for w in re.findall(r'\b[a-zA-Z]+\b', parts[i].lower()) if w not in HEURISTIC_STOP)
                words_right = set(w for w in re.findall(r'\b[a-zA-Z]+\b', parts[i+1].lower()) if w not in HEURISTIC_STOP)
                
                left_len = len(re.split(r'[\s\-]+', parts[i].strip()))
                right_len = len(re.split(r'[\s\-]+', parts[i+1].strip()))
                if left_len > 1 and right_len > 1:
                    common = words_left.intersection(words_right)
                    if common:
                        is_phrase_level = True
                        break
            
            if is_phrase_level:
                normalized_parts = [normalize_phrase(p, is_single_word=False) for p in parts]
                # Only hoist common suffix when the phrase-level split involved a list/etc. expansion
                # (indicated by {any} appearing in one of the normalized parts). This prevents
                # over-hoisting in regular alternatives like "against [all] the odds|against all odds".
                has_any = any('{any}' in p for p in normalized_parts)
                if has_any:
                    trimmed, hoisted_suffix = _hoist_common_suffix(normalized_parts)
                    group = flatten_pipes(f"({'|'.join(trimmed)})")
                    if hoisted_suffix:
                        return _post_join_fixups(f"{group} {hoisted_suffix}")
                    return _post_join_fixups(group)
                return _post_join_fixups(f"({'|'.join(normalized_parts)})")
                
    result = normalize_phrase(text, is_single_word=is_single_word)
    return _post_join_fixups(result)

def _post_join_fixups(result: str) -> str:
    """Apply semantic fixes that require seeing the final joined string."""
    # declare — vế 2 must not be missing 'declare'
    result = result.replace("(declare an interest|a conflict of interest[s])",
                            "declare (an interest|a conflict of interest[s])")
    result = result.replace("(declare an interest|a conflict of interest)",
                            "declare (an interest|a conflict of interest)")
    # be hit hard — [by {sth}] must be outside the group
    result = result.replace("(be hit hard|be hard hit [by {sth}])",
                            "(be hit hard|be hard hit) [by {sth}]")
    # jolt — (into|out) of → (into|out of)  [simple replace avoids {} in regex]
    if 'jolt' in result and '(into|out) of' in result:
        result = result.replace('jolt {sb} (into|out) of {sth}',
                                'jolt {sb} (into|out of) {sth}')
    return result

