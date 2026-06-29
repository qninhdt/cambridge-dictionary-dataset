"""
Cambridge Dictionary Word Crawler
===================================
Crawls full content for each word and stores in SQLite incrementally.
"""

import argparse
import json
import queue
import re
import sqlite3
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock
import requests
from src.utils.nlp import map_pos

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("Missing dependency: pip install beautifulsoup4 tqdm requests")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Missing dependency: pip install beautifulsoup4 tqdm requests")
    sys.exit(1)

BASE_URL = "https://dictionary.cambridge.org/dictionary/english/{word}"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
DELAY_PER_WORKER = 0.5   # seconds between requests per thread
MAX_RETRIES = 3

state_lock = Lock()
write_queue = queue.Queue()
thread_local = threading.local()

def get_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
        thread_local.session.headers.update(HEADERS)
    return thread_local.session

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS words (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    word         TEXT    UNIQUE NOT NULL,         -- URL slug  (e.g. "give-up")
    display_form TEXT,                            -- display   (e.g. "give up")
    entry_type   TEXT    DEFAULT 'word',          -- word | phrasal_verb | idiom | phrase | expression
    status       TEXT    DEFAULT 'pending',       -- pending | done | error | not_found
    crawled_at   TEXT,
    error_msg    TEXT
);

CREATE TABLE IF NOT EXISTS entries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    word_id         INTEGER NOT NULL REFERENCES words(id) ON DELETE CASCADE,
    entry_order     INTEGER DEFAULT 0,
    headword        TEXT,
    pos             TEXT,           -- noun, verb, adjective, adverb, etc.
    grammar         TEXT,           -- [T], [I], [C], [U], [ + to infinitive ], etc.
    pronunciation_uk TEXT,          -- IPA string
    pronunciation_us TEXT,          -- IPA string
    audio_uk_url    TEXT,
    audio_us_url    TEXT,
    dictionary_source TEXT          -- E.g. American Dictionary, Business English, etc.
);

CREATE TABLE IF NOT EXISTS entry_inflections (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id       INTEGER NOT NULL REFERENCES entries(id) ON DELETE CASCADE,
    form_type      TEXT,            -- e.g. "plural", "past tense"
    inflected_form TEXT NOT NULL    -- e.g. "children", "went"
);

CREATE INDEX IF NOT EXISTS idx_inflections_form ON entry_inflections(inflected_form);

CREATE TABLE IF NOT EXISTS senses (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id    INTEGER NOT NULL REFERENCES entries(id) ON DELETE CASCADE,
    sense_order INTEGER DEFAULT 0,
    guideword   TEXT,               -- guide word grouping this sense
    definition  TEXT NOT NULL,
    cefr_level  TEXT,               -- A1 A2 B1 B2 C1 C2
    grammar     TEXT,               -- sense-level grammar hint
    domain      TEXT,               -- COMPUTING, INSURANCE, MEDICAL, etc.
    labels      TEXT DEFAULT '[]',  -- JSON array: ["formal","literary","UK","old-fashioned"]
    phrase_title TEXT               -- If this sense is part of an inline phrase, its title
);

CREATE TABLE IF NOT EXISTS sense_synonyms (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    sense_id    INTEGER NOT NULL REFERENCES senses(id) ON DELETE CASCADE,
    synonym     TEXT NOT NULL,      -- e.g. "call off"
    slug        TEXT,               -- e.g. "call-off"
    is_antonym  INTEGER DEFAULT 0   -- 1 = antonym, 0 = synonym
);

CREATE INDEX IF NOT EXISTS idx_synonyms_sense ON sense_synonyms(sense_id);

CREATE TABLE IF NOT EXISTS examples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    sense_id    INTEGER NOT NULL REFERENCES senses(id) ON DELETE CASCADE,
    example_order INTEGER DEFAULT 0,
    example     TEXT NOT NULL,
    is_extra    INTEGER DEFAULT 0,    -- 1 = from "More examples" accordion
    collocation TEXT                  -- lu/collocation pattern if present
);

CREATE TABLE IF NOT EXISTS collocations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    word_id     INTEGER NOT NULL REFERENCES words(id) ON DELETE CASCADE,
    collocation TEXT NOT NULL,         -- e.g. "extended run"
    example     TEXT,                  -- sentence example
    source      TEXT                   -- source (e.g. Wikipedia)
);

CREATE INDEX IF NOT EXISTS idx_collocations_word ON collocations(word_id);

-- Cambridge SMART Vocabulary topic taxonomy
CREATE TABLE IF NOT EXISTS topics (
    id    INTEGER PRIMARY KEY AUTOINCREMENT,
    slug  TEXT UNIQUE NOT NULL,    -- e.g. "departing" (from URL path)
    title TEXT,                    -- e.g. "Departing"
    url   TEXT                     -- full topic URL
);

-- Many-to-many: which topic(s) does each word belong to?
CREATE TABLE IF NOT EXISTS word_topics (
    word_id  INTEGER NOT NULL REFERENCES words(id)  ON DELETE CASCADE,
    topic_id INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    PRIMARY KEY (word_id, topic_id)
);

CREATE INDEX IF NOT EXISTS idx_words_status   ON words(status);
CREATE INDEX IF NOT EXISTS idx_entries_word   ON entries(word_id);
CREATE INDEX IF NOT EXISTS idx_senses_entry   ON senses(entry_id);
CREATE INDEX IF NOT EXISTS idx_examples_sense ON examples(sense_id);
CREATE INDEX IF NOT EXISTS idx_word_topics_w  ON word_topics(word_id);
CREATE INDEX IF NOT EXISTS idx_word_topics_t  ON word_topics(topic_id);
"""

def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA cache_size=-10000;")
    conn.executescript(SCHEMA)
    for col_sql, label in [
        ("ALTER TABLE words ADD COLUMN entry_type   TEXT DEFAULT 'word'", "entry_type"),
        ("ALTER TABLE words ADD COLUMN display_form TEXT",                "display_form"),
        ("ALTER TABLE entries ADD COLUMN dictionary_source TEXT",          "dictionary_source"),
        ("ALTER TABLE senses ADD COLUMN phrase_title TEXT",               "phrase_title"),
    ]:
        try:
            conn.execute(col_sql)
            conn.commit()
        except Exception:
            pass
            
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS entry_inflections (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_id       INTEGER NOT NULL REFERENCES entries(id) ON DELETE CASCADE,
            form_type      TEXT,
            inflected_form TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_inflections_form ON entry_inflections(inflected_form);

        CREATE TABLE IF NOT EXISTS sense_synonyms (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            sense_id    INTEGER NOT NULL REFERENCES senses(id) ON DELETE CASCADE,
            synonym     TEXT NOT NULL,
            slug        TEXT,
            is_antonym  INTEGER DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_synonyms_sense ON sense_synonyms(sense_id);

        CREATE TABLE IF NOT EXISTS collocations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            word_id     INTEGER NOT NULL REFERENCES words(id) ON DELETE CASCADE,
            collocation TEXT NOT NULL,
            example     TEXT,
            source      TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_collocations_word ON collocations(word_id);

        CREATE TABLE IF NOT EXISTS word_alternatives (
            word_id          INTEGER NOT NULL REFERENCES words(id) ON DELETE CASCADE,
            alternative_word TEXT NOT NULL,
            alternative_type TEXT NOT NULL,
            PRIMARY KEY (word_id, alternative_word)
        );
        CREATE INDEX IF NOT EXISTS idx_word_alternatives_alt ON word_alternatives(alternative_word);

        CREATE TABLE IF NOT EXISTS temp_redirects (
            word_id    INTEGER NOT NULL REFERENCES words(id) ON DELETE CASCADE,
            pos        TEXT,
            definition TEXT
        );
    """)
    conn.commit()
    return conn

def load_words(conn: sqlite3.Connection, words_file: str):
    rows: list[tuple[str, str, str]] = []
    with open(words_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                rows.append((parts[0], parts[1], parts[2]))
            elif len(parts) == 2:
                rows.append((parts[0], parts[0], parts[1]))
            else:
                rows.append((parts[0], parts[0], "word"))

    cur = conn.cursor()
    cur.executemany(
        "INSERT OR IGNORE INTO words (word, display_form, entry_type) VALUES (?, ?, ?)",
        rows,
    )
    cur.executemany(
        """UPDATE words
           SET display_form = COALESCE(NULLIF(display_form, word), ?),
               entry_type   = ?
           WHERE word = ?""",
        [(display, etype, slug) for slug, display, etype in rows],
    )
    conn.commit()
    total = cur.execute("SELECT COUNT(*) FROM words").fetchone()[0]
    type_counts = dict(cur.execute(
        "SELECT entry_type, COUNT(*) FROM words GROUP BY entry_type"
    ).fetchall())
    print(f"  Loaded {len(rows)} entries → {total} total in DB")
    for t, n in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t:<16}: {n}")

def get_pending(conn: sqlite3.Connection) -> list[tuple[int, str, str]]:
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT id, word, entry_type FROM words WHERE status='pending' ORDER BY id"
    ).fetchall()
    return rows

def matches_any_pattern(defn):
    if not defn:
        return False
        
    s = defn.strip()
    s = re.sub(r'\s+', ' ', s)
    s_lower = s.lower()
    
    # Check grammatical inflections first (fnmatch patterns)
    inflection_patterns = [
        'past simple of *',
        'past simple and past participle of *',
        'past participle of *',
        'past participle, past simple of *',
        'present participle of *',
        'plural of *',
        'pl of *',
        'comparative of *',
        'superlative of *'
    ]
    import fnmatch
    for pat in inflection_patterns:
        if fnmatch.fnmatch(s_lower, pat):
            return True
            
    # Check spelling redirects regex
    spelling_pat = re.compile(
        r"^(?:the |a |an |mainly |old-fashioned |older |old |non-standard |US and Australian English |written |informal |)*?"
        r"(?:US|UK|English|Australian|non-standard|old-fashioned|older|old|)*?"
        r"\s*spelling\s+of\s+(.+)$",
        re.IGNORECASE
    )
    if spelling_pat.match(s):
        if "computer program" not in s_lower:
            return True
            
    # Check abbreviation redirects
    abbrev_pat = re.compile(
        r"^(?:the |a |an |mainly |old-fashioned |older |old |non-standard |UK |US |written |informal |offensive |)*?"
        r"abbreviation\s+(?:for|of)\s+(.+)$",
        re.IGNORECASE
    )
    if abbrev_pat.match(s):
        if "consisting of" not in s_lower and "that consists" not in s_lower:
            return True
            
    # Check short form redirects
    sf_pat = re.compile(
        r"^(?:the |a |an |mainly |old-fashioned |older |old |non-standard |)*?"
        r"short\s+form\s+(?:of|for)\s+(.+)$",
        re.IGNORECASE
    )
    if sf_pat.match(s):
        if "giving only" not in s_lower and "combination of words" not in s_lower:
            return True
            
    # Check arrow redirects
    if s.startswith('→ '):
        return True
        
    # Check another word / another spelling redirects
    if s_lower.startswith('another spelling of ') or s_lower.startswith('another word for '):
        return True
        
    return False

class DbWriterThread(threading.Thread):
    def __init__(self, db_path: str, batch_size: int = 20):
        super().__init__()
        self.db_path = db_path
        self.batch_size = batch_size
        self.daemon = True
        self.running = True

    def run(self):
        conn = init_db(self.db_path)
        batch = []
        last_flush_time = time.time()
        FLUSH_INTERVAL = 30

        while self.running or not write_queue.empty():
            try:
                item = write_queue.get(timeout=0.5)
                if item is None:
                    write_queue.task_done()
                    break
                batch.append(item)
                write_queue.task_done()
            except queue.Empty:
                pass

            now = time.time()
            should_flush = (
                len(batch) >= self.batch_size
                or (len(batch) > 0 and write_queue.empty())
                or (len(batch) > 0 and (now - last_flush_time) >= FLUSH_INTERVAL)
            )
            if should_flush:
                self.write_batch(conn, batch)
                batch = []
                last_flush_time = time.time()

        if batch:
            self.write_batch(conn, batch)
        conn.close()

    def write_batch(self, conn: sqlite3.Connection, batch: list[tuple]):
        cur = conn.cursor()
        now = datetime.utcnow().isoformat()
        try:
            for word_id, entries_data, status, error_msg, collocations in batch:
                cur.execute(
                    "UPDATE words SET status=?, crawled_at=?, error_msg=? WHERE id=?",
                    (status, now, error_msg, word_id),
                )
                cur.execute("DELETE FROM entries WHERE word_id=?", (word_id,))
                cur.execute("DELETE FROM collocations WHERE word_id=?", (word_id,))

                for entry_data in entries_data:
                    cur.execute(
                        """INSERT INTO entries
                           (word_id, entry_order, headword, pos, grammar,
                            pronunciation_uk, pronunciation_us, audio_uk_url, audio_us_url,
                            dictionary_source)
                           VALUES (?,?,?,?,?,?,?,?,?,?)""",
                        (
                            word_id,
                            entry_data["entry_order"],
                            entry_data["headword"],
                            entry_data["pos"],
                            entry_data["grammar"],
                            entry_data["pronunciation_uk"],
                            entry_data["pronunciation_us"],
                            entry_data["audio_uk_url"],
                            entry_data["audio_us_url"],
                            entry_data.get("dictionary_source", ""),
                        ),
                    )
                    entry_id = cur.lastrowid

                    infls = entry_data.get("inflections", [])
                    if infls:
                        cur.executemany(
                            """INSERT INTO entry_inflections (entry_id, form_type, inflected_form)
                               VALUES (?,?,?)""",
                            [(entry_id, inf["form_type"], inf["inflected_form"]) for inf in infls],
                        )

                    for sense_data in entry_data["senses"]:
                        defn = sense_data["definition"]
                        if matches_any_pattern(defn):
                            cur.execute(
                                """INSERT INTO temp_redirects (word_id, pos, definition)
                                   VALUES (?,?,?)""",
                                (word_id, entry_data["pos"], defn),
                            )
                            continue

                        cur.execute(
                            """INSERT INTO senses
                               (entry_id, sense_order, guideword, definition,
                                cefr_level, grammar, domain, labels, phrase_title)
                               VALUES (?,?,?,?,?,?,?,?,?)""",
                            (
                                entry_id,
                                sense_data["sense_order"],
                                sense_data["guideword"],
                                defn,
                                sense_data["cefr_level"],
                                sense_data["grammar"],
                                sense_data["domain"],
                                json.dumps(sense_data["labels"], ensure_ascii=False),
                                sense_data.get("phrase_title"),
                            ),
                        )
                        sense_id = cur.lastrowid

                        syns = sense_data.get("synonyms", [])
                        if syns:
                            cur.executemany(
                                """INSERT INTO sense_synonyms (sense_id, synonym, slug, is_antonym)
                                   VALUES (?,?,?,?)""",
                                [(sense_id, syn["synonym"], syn["slug"], syn["is_antonym"]) for syn in syns],
                            )

                        exs = sense_data["examples"]
                        if exs:
                            cur.executemany(
                                """INSERT INTO examples
                                   (sense_id, example_order, example, collocation, is_extra)
                                   VALUES (?,?,?,?,?)""",
                                [(sense_id, ex_i, ex_data["text"], ex_data["collocation"], ex_data.get("is_extra", 0))
                                 for ex_i, ex_data in enumerate(exs)],
                            )

                if collocations:
                    cur.executemany(
                        """INSERT INTO collocations (word_id, collocation, example, source)
                           VALUES (?,?,?,?)""",
                        [(word_id, c["collocation"], c["example"], c["source"]) for c in collocations],
                    )

                all_rel_slugs = []
                topic_links = []

                for topic_data in entries_data[0].get("topics", []) if entries_data else []:
                    cur.execute(
                        "INSERT OR IGNORE INTO topics (slug, title, url) VALUES (?,?,?)",
                        (topic_data["slug"], topic_data["title"], topic_data["url"]),
                    )
                    cur.execute("SELECT id FROM topics WHERE slug=?", (topic_data["slug"],))
                    topic_id = cur.fetchone()[0]

                    cur.execute(
                        "INSERT OR IGNORE INTO word_topics (word_id, topic_id) VALUES (?,?)",
                        (word_id, topic_id),
                    )

                    for rel_slug in topic_data.get("related_slugs", []):
                        all_rel_slugs.append(rel_slug)
                        topic_links.append((rel_slug, topic_id))

                more_meanings = entries_data[0].get("more_meanings", []) if entries_data else []

                all_new_words = list(set(all_rel_slugs + more_meanings))
                if all_new_words:
                    cur.executemany(
                        "INSERT OR IGNORE INTO words (word, status) VALUES (?, 'seen')",
                        [(w,) for w in all_new_words]
                    )

                if topic_links:
                    unique_rel_slugs = list(set(all_rel_slugs))
                    slug_to_id = {}
                    for i in range(0, len(unique_rel_slugs), 900):
                        chunk = unique_rel_slugs[i : i + 900]
                        placeholders = ",".join("?" for _ in chunk)
                        cur.execute(
                            f"SELECT word, id FROM words WHERE word IN ({placeholders})",
                            chunk,
                        )
                        for w, wid in cur.fetchall():
                            slug_to_id[w] = wid

                    word_topic_inserts = []
                    for rel_slug, topic_id in topic_links:
                        rel_word_id = slug_to_id.get(rel_slug)
                        if rel_word_id:
                            word_topic_inserts.append((rel_word_id, topic_id))

                    if word_topic_inserts:
                        cur.executemany(
                            "INSERT OR IGNORE INTO word_topics (word_id, topic_id) VALUES (?,?)",
                            word_topic_inserts
                        )

            conn.commit()
            tqdm.write(f"\n[INFO] Successfully committed batch of {len(batch)} words to DB.")
        except Exception as e:
            conn.rollback()
            tqdm.write(f"\n[ERROR] Database write batch failed: {e}")

def fetch(url: str) -> tuple[str, int]:
    session = get_session()
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, timeout=20)
            status_code = resp.status_code
            if status_code == 200:
                return resp.text, 200
            elif status_code == 404:
                return "", 404
            elif status_code == 429:
                tqdm.write(f"\n[WARNING] Rate limited (429) on {url}. Retrying...")
                time.sleep(10 * (attempt + 1))
            elif status_code == 403:
                tqdm.write(f"\n[WARNING] Forbidden (403) on {url}. Possible IP block.")
                time.sleep(5 * (attempt + 1))
            else:
                time.sleep(2 ** attempt)
        except requests.exceptions.RequestException as e:
            tqdm.write(f"\n[WARNING] Network error ({e.__class__.__name__}) on {url}. Retrying...")
            time.sleep(2 ** attempt)
        except Exception as e:
            tqdm.write(f"\n[WARNING] Unexpected fetch error ({e}) on {url}. Retrying...")
            time.sleep(2 ** attempt)
    return "", 0

def _text(tag) -> str:
    if tag is None:
        return ""
    return tag.get_text(" ", strip=True)

def _get_audio_url(block, region: str) -> str:
    pron = block.select_one(f".{region}.dpron-i .daud")
    if not pron:
        return ""
    src = pron.select_one("source[type='audio/mpeg']")
    if not src:
        src = pron.select_one("source")
    if src:
        url = src.get("src", "")
        if url.startswith("//"):
            url = "https:" + url
        return url
    return ""

def parse_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    entries = []
    entry_blocks = soup.select(".entry-body__el")
    if not entry_blocks:
        entry_blocks = soup.select(".di-body")

    for entry_idx, block in enumerate(entry_blocks):
        entry: dict = {
            "entry_order": entry_idx,
            "headword": "",
            "pos": "",
            "grammar": "",
            "pronunciation_uk": "",
            "pronunciation_us": "",
            "audio_uk_url": "",
            "audio_us_url": "",
            "dictionary_source": "",
            "inflections": [],
            "senses": [],
        }

        hw = block.select_one(".hw, .dhw, .headword")
        entry["headword"] = _text(hw)

        pos_tag = block.select_one(".pos-header .pos.dpos, .di-info .pos.dpos, .pos.dpos")
        entry["pos"] = _text(pos_tag)

        gram_tag = block.select_one(".pos-header .posgram .gram.dgram, .di-info .gram.dgram, .posgram .gram.dgram")
        if gram_tag:
            gc = gram_tag.select(".gc.dgc")
            if gc:
                entry["grammar"] = "[ " + " ".join(_text(g) for g in gc) + " ]"
            else:
                entry["grammar"] = _text(gram_tag)

        uk_ipa = block.select_one(".pos-header .uk.dpron-i .ipa.dipa, .uk.dpron-i .ipa.dipa, .dpron-i.uk .ipa")
        entry["pronunciation_uk"] = _text(uk_ipa)

        us_ipa = block.select_one(".pos-header .us.dpron-i .ipa.dipa, .us.dpron-i .ipa.dipa, .dpron-i.us .ipa")
        entry["pronunciation_us"] = _text(us_ipa)

        entry["audio_uk_url"] = _get_audio_url(block, "uk")
        entry["audio_us_url"] = _get_audio_url(block, "us")

        infl_blocks = block.select(".irreg-infls .inf-group, .dinfls .dinfg")
        if infl_blocks:
            for infl_block in infl_blocks:
                lab_tag = infl_block.select_one(".lab, .inf-lab")
                inf_tag = infl_block.select_one(".inf")
                if inf_tag:
                    form_type = _text(lab_tag).strip()
                    inflected_form = _text(inf_tag).strip()
                    if inflected_form and not (inflected_form.startswith("-") and inflected_form.endswith("-")):
                        entry["inflections"].append({
                            "form_type": form_type,
                            "inflected_form": inflected_form
                        })
        else:
            for inf_tag in block.select(".irreg-infls .inf, .dinfls .inf"):
                inf_text = _text(inf_tag).strip()
                if inf_text and not (inf_text.startswith("-") and inf_text.endswith("-")):
                    entry["inflections"].append({
                        "form_type": "",
                        "inflected_form": inf_text
                    })

        curr = block
        dict_container = None
        while curr:
            if curr.get("class") and "dictionary" in curr.get("class"):
                dict_container = curr
                break
            curr = curr.parent
                
        if dict_container:
            header = dict_container.select_one("h2.c_hh, .di-title")
            if header:
                src_text = header.text.strip()
                if src_text.lower() == entry["headword"].lower():
                    entry["dictionary_source"] = "Cambridge Advanced Learner's Dictionary"
                else:
                    entry["dictionary_source"] = src_text

        sense_order = 0
        sense_blocks = block.select(".dsense")

        if not sense_blocks:
            def_blocks = block.select(".ddef_block")
            for def_block in def_blocks:
                sense: dict = {
                    "sense_order": sense_order,
                    "guideword": "",
                    "definition": "",
                    "cefr_level": "",
                    "grammar": "",
                    "domain": "",
                    "labels": [],
                    "phrase_title": None,
                    "synonyms": [],
                    "examples": [],
                }

                def_tag = def_block.select_one(".def.ddef_d")
                if def_tag:
                    sense["definition"] = _text(def_tag).rstrip(":").strip()

                if not sense["definition"]:
                    continue

                cefr_tag = def_block.select_one(".epp-xref")
                sense["cefr_level"] = _text(cefr_tag)

                gram_tag = def_block.select_one(".ddef_h .gram.dgram")
                if gram_tag:
                    gc = gram_tag.select(".gc.dgc")
                    sense["grammar"] = (
                        "[ " + " ".join(_text(g) for g in gc) + " ]"
                        if gc else _text(gram_tag)
                    )

                domain_tag = def_block.select_one(".domain.ddomain")
                sense["domain"] = _text(domain_tag)

                lab_tags = def_block.select(".lab.dlab .usage.dusage")
                sense["labels"] = [_text(l) for l in lab_tags if _text(l)]

                phrase_parent = def_block.find_parent(class_="phrase-block") or def_block.find_parent(class_="idiom-block")
                if phrase_parent:
                    phrase_title_tag = phrase_parent.select_one(".phrase-title, .di-title")
                    if phrase_title_tag:
                        sense["phrase_title"] = phrase_title_tag.text.strip()

                for a in def_block.select("a[href*='/thesaurus/']"):
                    href = a.get("href", "")
                    if "articles" in href:
                        continue
                    path = href.split("?")[0].rstrip("/")
                    slug = path.split("/")[-1]
                    
                    is_antonym = 0
                    sibling = a.find_previous(["span", "div", "h3", "h4"])
                    if sibling and any(k in sibling.text.lower() for k in ["opposite", "antonym"]):
                        is_antonym = 1
                        
                    sense["synonyms"].append({
                        "synonym": a.text.strip(),
                        "slug": slug,
                        "is_antonym": is_antonym
                    })

                for ex_block in def_block.select(".examp.dexamp"):
                    eg_tag = ex_block.select_one(".eg.deg")
                    if not eg_tag:
                        continue
                    lu_tag = ex_block.select_one(".lu.dlu")
                    sense["examples"].append({
                        "text": _text(eg_tag),
                        "collocation": _text(lu_tag),
                        "is_extra": 0,
                    })

                sense_order += 1
                entry["senses"].append(sense)
        else:
            for sense_block in sense_blocks:
                gw_tag = sense_block.select_one(".guideword.dsense_gw")
                guideword = _text(gw_tag).strip("()")

                def_blocks = sense_block.select(".ddef_block")
                for def_block in def_blocks:
                    sense: dict = {
                        "sense_order": sense_order,
                        "guideword": guideword,
                        "definition": "",
                        "cefr_level": "",
                        "grammar": "",
                        "domain": "",
                        "labels": [],
                        "phrase_title": None,
                        "synonyms": [],
                        "examples": [],
                    }

                    def_tag = def_block.select_one(".def.ddef_d")
                    if def_tag:
                        sense["definition"] = _text(def_tag).rstrip(":").strip()

                    if not sense["definition"]:
                        continue

                    cefr_tag = def_block.select_one(".epp-xref")
                    sense["cefr_level"] = _text(cefr_tag)

                    gram_tag = def_block.select_one(".ddef_h .gram.dgram")
                    if gram_tag:
                        gc = gram_tag.select(".gc.dgc")
                        sense["grammar"] = (
                            "[ " + " ".join(_text(g) for g in gc) + " ]"
                            if gc else _text(gram_tag)
                        )

                    domain_tag = def_block.select_one(".domain.ddomain")
                    sense["domain"] = _text(domain_tag)

                    lab_tags = def_block.select(".lab.dlab .usage.dusage")
                    sense["labels"] = [_text(l) for l in lab_tags if _text(l)]

                    phrase_parent = def_block.find_parent(class_="phrase-block") or def_block.find_parent(class_="idiom-block")
                    if phrase_parent:
                        phrase_title_tag = phrase_parent.select_one(".phrase-title, .di-title")
                        if phrase_title_tag:
                            sense["phrase_title"] = phrase_title_tag.text.strip()

                    for a in def_block.select("a[href*='/thesaurus/']"):
                        href = a.get("href", "")
                        if "articles" in href:
                            continue
                        path = href.split("?")[0].rstrip("/")
                        slug = path.split("/")[-1]
                        
                        is_antonym = 0
                        sibling = a.find_previous(["span", "div", "h3", "h4"])
                        if sibling and any(k in sibling.text.lower() for k in ["opposite", "antonym"]):
                            is_antonym = 1
                            
                        sense["synonyms"].append({
                            "synonym": a.text.strip(),
                            "slug": slug,
                            "is_antonym": is_antonym
                        })

                    for ex_block in def_block.select(".examp.dexamp"):
                        eg_tag = ex_block.select_one(".eg.deg")
                        if not eg_tag:
                            continue
                        lu_tag = ex_block.select_one(".lu.dlu")
                        sense["examples"].append({
                            "text": _text(eg_tag),
                            "collocation": _text(lu_tag),
                            "is_extra": 0,
                        })

                    seen_texts = {ex["text"] for ex in sense["examples"]}
                    for daccord in sense_block.select(".daccord"):
                        if "smartt" in (daccord.get("class") or []):
                            continue
                            
                        title_tag = daccord.select_one(".daccord_lt")
                        title_text = title_tag.text.lower() if title_tag else ""
                        if "thesaurus" in title_text or "synonym" in title_text or "opposite" in title_text:
                            for a in daccord.select("a[href*='/thesaurus/']"):
                                href = a.get("href", "")
                                if "articles" in href:
                                    continue
                                path = href.split("?")[0].rstrip("/")
                                slug = path.split("/")[-1]
                                is_antonym = 1 if "opposite" in title_text or "antonym" in title_text else 0
                                
                                sense["synonyms"].append({
                                    "synonym": a.text.strip(),
                                    "slug": slug,
                                    "is_antonym": is_antonym
                                })
                            continue
                            
                        for li in daccord.select("li.eg.dexamp.hax"):
                            text = _text(li)
                            if text and text not in seen_texts:
                                sense["examples"].append({
                                    "text": text,
                                    "collocation": "",
                                    "is_extra": 1,
                                })
                                seen_texts.add(text)

                    sense_order += 1
                    entry["senses"].append(sense)

        if entry["headword"] or entry["pos"] or entry["senses"]:
            entries.append(entry)

    topics: list[dict] = []
    for smart_block in soup.select(".smartt.daccord"):
        topic_link = smart_block.select_one(".daccord_lt a")
        if not topic_link:
            continue
        topic_url = topic_link.get("href", "").strip()
        topic_title = _text(topic_link)
        slug_match = re.search(r'/topics/[^/]+/([^/?]+)/?', topic_url)
        if not slug_match:
            slug_match = re.search(r'/([^/?]+)/?$', topic_url)
        topic_slug = slug_match.group(1) if slug_match else topic_url

        related_slugs: list[str] = []
        for a in smart_block.select(".daccord_lb a[href*='/dictionary/english/']"):
            href = a.get("href", "")
            m = re.search(r'/dictionary/english/([^?\s]+)', href)
            if m:
                related_slugs.append(m.group(1).strip())

        topics.append({
            "slug": topic_slug,
            "title": topic_title,
            "url": topic_url,
            "related_slugs": related_slugs,
        })

    more_meanings_slugs = []
    for container in soup.select(".cdo-more-results"):
        for a in container.select("a[href*='/dictionary/english/']"):
            href = a.get("href", "")
            m = re.search(r'/dictionary/english/([^?\s]+)', href)
            if m:
                more_meanings_slugs.append(m.group(1).strip())

    has_collocations = False
    if soup.select_one("a[href*='/collocation/']"):
        has_collocations = True

    if entries:
        entries[0]["topics"] = topics
        entries[0]["more_meanings"] = more_meanings_slugs
        entries[0]["has_collocations"] = has_collocations

    return entries

def parse_collocation_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    collocations = []
    
    col_blocks = soup.select("div.eg.lbb.lb-cm.lpt-10")
    for cb in col_blocks:
        col_a = cb.select_one("a.hdib.tb.lmb-10")
        if not col_a:
            continue
        col_text = col_a.text.strip()
        
        ex_div = cb.select_one("div.dexamp")
        ex_text = ex_div.text.strip() if ex_div else ""
        
        src_div = cb.select_one(".dsource_e")
        src_text = src_div.text.strip() if src_div else ""
        if not src_text:
            dsource = cb.select_one(".dsource")
            if dsource:
                src_text = dsource.text.replace("From", "").replace("the", "").strip()
                
        collocations.append({
            "collocation": col_text,
            "example": ex_text,
            "source": src_text
        })
    return collocations

def process_word(
    word_id: int,
    word: str,
    entry_type: str,
) -> str:
    time.sleep(DELAY_PER_WORKER)
    url = BASE_URL.format(word=word)
    html, code = fetch(url)

    if code == 404 or (code == 200 and not html):
        write_queue.put((word_id, [], "not_found", None, None))
        return "not_found"

    if code != 200:
        write_queue.put((word_id, [], "pending", f"HTTP Error status: {code}", None))
        return "error"

    try:
        entries = parse_page(html)
    except Exception as e:
        write_queue.put((word_id, [], "error", f"Parse error: {e}", None))
        return "error"

    if not entries:
        write_queue.put((word_id, [], "not_found", "No entries found", None))
        return "not_found"

    collocations = []
    has_collocations = entries[0].get("has_collocations", False) if entries else False
    if entry_type == "word" and has_collocations:
        time.sleep(DELAY_PER_WORKER)
        colloc_url = f"https://dictionary.cambridge.org/collocation/english/{word}"
        colloc_html, colloc_code = fetch(colloc_url)
        
        if colloc_code == 200 and colloc_html:
            try:
                collocations = parse_collocation_page(colloc_html)
            except Exception:
                pass

    write_queue.put((word_id, entries, "done", None, collocations))
    return "done"

def print_stats(conn: sqlite3.Connection):
    cur = conn.cursor()
    rows = cur.execute("SELECT status, COUNT(*) FROM words GROUP BY status").fetchall()
    stats = {r[0]: r[1] for r in rows}
    total = sum(stats.values())
    done = stats.get("done", 0)
    pct = done / total * 100 if total else 0
    print(f"\n── Progress ───────────────────────────")
    print(f"  Total    : {total}")
    print(f"  Done     : {done}  ({pct:.1f}%)")
    print(f"  Pending  : {stats.get('pending', 0)}")
    print(f"  Not found: {stats.get('not_found', 0)}")
    print(f"  Errors   : {stats.get('error', 0)}")
    
    type_rows = cur.execute(
        "SELECT entry_type, COUNT(*), SUM(status='done') FROM words GROUP BY entry_type"
    ).fetchall()
    print(f"\n── By Type ────────────────────────────")
    for etype, cnt, dcnt in sorted(type_rows, key=lambda x: -x[1]):
        print(f"  {etype:<16}: {cnt:>6}  (done: {dcnt or 0})")
    print(f"───────────────────────────────────────")

def crawl_dictionary(words_file: str = None, db_path: str = "data/cambridge.db", workers: int = 5, load_only: bool = False, stats_only: bool = False):
    conn = init_db(db_path)

    if stats_only:
        print_stats(conn)
        conn.close()
        return

    if words_file:
        print(f"\n[1/2] Loading words from: {words_file}")
        load_words(conn, words_file)

    if load_only:
        print_stats(conn)
        conn.close()
        return

    pending = get_pending(conn)
    conn.close()

    if not pending:
        conn = init_db(db_path)
        print("\nNo pending words to crawl.")
        print_stats(conn)
        conn.close()
        return

    print(f"\n[2/2] Crawling {len(pending)} pending words with {workers} workers...")
    print()

    writer = DbWriterThread(db_path, batch_size=20)
    writer.start()

    counters = {"done": 0, "error": 0, "not_found": 0}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_word, wid, word, etype): word
            for wid, word, etype in pending
        }

        with tqdm(
            total=len(pending),
            unit="word",
            dynamic_ncols=True,
            smoothing=0.0,
            mininterval=1.5,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ) as pbar:
            for future in as_completed(futures):
                word = futures[future]
                try:
                    status = future.result()
                except Exception:
                    status = "error"

                counters[status] = counters.get(status, 0) + 1
                pbar.set_postfix(
                    done=counters["done"],
                    err=counters["error"],
                    refresh=False,
                )
                pbar.update(1)

    writer.running = False
    write_queue.put(None)
    writer.join()

    conn = init_db(db_path)
    print()
    clean_alternatives(conn)
    print_stats(conn)
    conn.close()
    print("\nDone!")

def clean_alternatives(conn: sqlite3.Connection):
    import re
    print("Post-processing: cleaning spelling alternatives (US-first)...")
    cursor = conn.cursor()
    
    # Pre-cache word ID lookups with HTML unescaping and normalization
    import html
    cursor.execute("SELECT id, word, display_form FROM words")
    word_id_cache = {}
    for r_id, word_slug, display in cursor.fetchall():
        if word_slug:
            slug_unesc = html.unescape(word_slug).lower().strip()
            word_id_cache[slug_unesc] = r_id
            word_id_cache[slug_unesc.replace("-", " ")] = r_id
        if display:
            disp_unesc = html.unescape(display).lower().strip()
            word_id_cache[disp_unesc] = r_id
            word_id_cache[disp_unesc.replace(" ", "-")] = r_id
            
    def check_cache(target):
        target_lower = target.lower()
        if target_lower in word_id_cache:
            return target
        t_hyphen = target_lower.replace(" ", "-")
        if t_hyphen in word_id_cache:
            return target
        if target_lower.startswith("the "):
            t_strip = target[4:].strip()
            t_strip_lower = t_strip.lower()
            if t_strip_lower in word_id_cache:
                return t_strip
            if t_strip_lower.replace(" ", "-") in word_id_cache:
                return t_strip
        return None

    def clean_target(target):
        target = re.split(r'[:;,]|\bor\b', target)[0].strip()
        target = target.strip('"\' ')
        
        suffixes = [
            ' mainly disapproving', ' mainly approving', ' mainly informal', ' mainly formal',
            ' uk old-fashioned', ' us old-fashioned', ' mainly uk', ' mainly us',
            ' uk specialized', ' us specialized', ' old-fashioned',
            ' informal', ' formal', ' specialized', ' disapproving', ' approving', 
            ' humorous', ' slang', ' literary', ' trademark', ' old use',
            ' Indian English', ' Australian English', ' Irish English', ' uk', ' us', ' old',
            ' noun', ' verb', ' adjective', ' adverb', ' s'
        ]
        suffixes.sort(key=len, reverse=True)
        
        changed = True
        while changed:
            changed = False
            target_lower = target.lower()
            res = check_cache(target)
            if res:
                return res
                
            for suffix in suffixes:
                if target_lower.endswith(suffix.lower()):
                    target = target[:-len(suffix)].strip()
                    changed = True
                    break
                    
        return check_cache(target)

    def parse_redirect_definition(defn):
        if not defn:
            return None
            
        s = defn.strip()
        s = re.sub(r'\s+', ' ', s)
        s_lower = s.lower()
        
        # 1. Check spelling redirects
        spelling_pat = re.compile(
            r"^(?:the |a |an |mainly |old-fashioned |older |old |non-standard |US and Australian English |written |informal |)*?"
            r"(?:US|UK|English|Australian|non-standard|old-fashioned|older|old|)*?"
            r"\s*spelling\s+of\s+(.+)$",
            re.IGNORECASE
        )
        m = spelling_pat.match(s)
        if m:
            target = clean_target(m.group(1))
            if target and "computer program" not in s_lower:
                return {
                    'type': 'UK spelling',
                    'target': target
                }
                
        # 2. Check abbreviation redirects
        abbrev_pat = re.compile(
            r"^(?:the |a |an |mainly |old-fashioned |older |old |non-standard |UK |US |written |informal |offensive |)*?"
            r"abbreviation\s+(?:for|of)\s+(.+)$",
            re.IGNORECASE
        )
        m = abbrev_pat.match(s)
        if m:
            target = clean_target(m.group(1))
            if target and "consisting of" not in s_lower and "that consists" not in s_lower:
                return {
                    'type': 'abbreviation',
                    'target': target
                }
                
        # 3. Check short form redirects
        sf_pat = re.compile(
            r"^(?:the |a |an |mainly |old-fashioned |older |old |non-standard |)*?"
            r"short\s+form\s+(?:of|for)\s+(.+)$",
            re.IGNORECASE
        )
        m = sf_pat.match(s)
        if m:
            target = clean_target(m.group(1))
            if target and "giving only" not in s_lower and "combination of words" not in s_lower:
                return {
                    'type': 'short form',
                    'target': target
                }
                
        # 4. Check arrow redirects (Unconditional match)
        if s.startswith('→ '):
            target = s[2:].strip()
            target = re.split(r'[:;,]|\bor\b', target)[0].strip()
            target = target.strip('"\' ')
            cleaned = clean_target(target)
            return {
                'type': 'arrow',
                'target': cleaned if cleaned else target
            }
            
        # 5. Check another word / another spelling redirects
        if s_lower.startswith('another spelling of '):
            target = clean_target(s[len('another spelling of '):])
            if target:
                return {
                    'type': 'another spelling',
                    'target': target
                }
        if s_lower.startswith('another word for '):
            target = clean_target(s[len('another word for '):])
            if target:
                return {
                    'type': 'another word',
                    'target': target
                }
                
        return None

    # Fetch all temp_redirects rows to process
    cursor.execute("""
        SELECT w.display_form, tr.pos, tr.definition
        FROM temp_redirects tr
        JOIN words w ON w.id = tr.word_id
    """)
    temp_rows = cursor.fetchall()

    # 1. Extract UK-to-US spelling mappings
    uk_to_us = {} # (uk_word, pos_char) -> us_word
    for headword, pos, definition in temp_rows:
        pos_char = map_pos(pos)
        if not pos_char:
            continue
        res = parse_redirect_definition(definition)
        if res and res['type'] == 'UK spelling':
            defn_lower = definition.lower()
            spelling_part = defn_lower.split("spelling")[0]
            if "us" in spelling_part:
                uk_word = res['target']
                us_word = headword
                uk_to_us[(uk_word.lower().strip(), pos_char)] = us_word.lower().strip()
            elif "uk" in spelling_part:
                uk_word = headword
                us_word = res['target']
                uk_to_us[(uk_word.lower().strip(), pos_char)] = us_word.lower().strip()

    # 2. Redirect UK spelling entries to US entries
    redirect_count = 0
    rename_count = 0
    for (uk_word, pos_char), us_word in uk_to_us.items():
        cursor.execute("SELECT id FROM words WHERE word = ? OR display_form = ?", (uk_word, uk_word))
        uk_row = cursor.fetchone()
        cursor.execute("SELECT id FROM words WHERE word = ? OR display_form = ?", (us_word, us_word))
        us_row = cursor.fetchone()
        
        if uk_row and us_row:
            uk_word_id = uk_row[0]
            us_word_id = us_row[0]
            
            cursor.execute("""
                DELETE FROM entries 
                WHERE word_id = ? 
                  AND id NOT IN (SELECT DISTINCT entry_id FROM senses)
            """, (us_word_id,))
                
            cursor.execute("""
                UPDATE entries 
                SET word_id = ? 
                WHERE word_id = ?
            """, (us_word_id, uk_word_id))
            redirect_count += 1
        elif uk_row and not us_row:
            uk_word_id = uk_row[0]
            us_slug = us_word.replace(" ", "-")
            cursor.execute("""
                UPDATE words
                SET word = ?, display_form = ?
                WHERE id = ?
            """, (us_slug, us_word, uk_word_id))
            rename_count += 1
    conn.commit()

    # 3. Populate word_alternatives table
    alternatives_to_insert = []
    for headword, pos, definition in temp_rows:
        pos_char = map_pos(pos)
        res = parse_redirect_definition(definition)
        if res:
            headword_clean = headword.lower().strip()
            target_clean = res['target'].lower().strip()
            
            if res['type'] == 'UK spelling':
                defn_lower = definition.lower()
                if "us" in defn_lower.split("spelling")[0]:
                    base_word = headword_clean
                    alt_word = target_clean
                else:
                    base_word = target_clean
                    alt_word = headword_clean
            else:
                base_word = target_clean
                alt_word = headword_clean
                
            if pos_char:
                base_word = uk_to_us.get((base_word, pos_char), base_word)
                
            base_word_id = word_id_cache.get(base_word)
            if base_word_id:
                alternatives_to_insert.append((base_word_id, alt_word, res['type']))

    cursor.executemany("""
        INSERT OR IGNORE INTO word_alternatives (word_id, alternative_word, alternative_type)
        VALUES (?, ?, ?)
    """, alternatives_to_insert)
    conn.commit()

    # 4. Drop temporary redirects table
    cursor.execute("DROP TABLE IF EXISTS temp_redirects")
    conn.commit()

    # 5. Clean up orphaned entries (entries with no remaining senses)
    cursor.execute("""
        DELETE FROM entries 
        WHERE id NOT IN (SELECT DISTINCT entry_id FROM senses)
    """)
    deleted_entries = cursor.rowcount
    conn.commit()

    print(f"  - Redirected {redirect_count} UK entries to US word IDs.")
    print(f"  - Renamed {rename_count} missing US words.")
    print(f"  - Populated {len(alternatives_to_insert)} rows in word_alternatives.")
    print(f"  - Deleted {deleted_entries} orphaned entries.")
