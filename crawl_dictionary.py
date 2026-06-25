#!/usr/bin/env python3
"""
Cambridge Dictionary Word Crawler
===================================
Crawls full content for each word and stores in SQLite incrementally.

Usage:
    python crawl_dictionary.py --words cambridge_words.txt --db cambridge.db
    python crawl_dictionary.py --words cambridge_words.txt --db cambridge.db --workers 5
    python crawl_dictionary.py --db cambridge.db --resume          # resume from last run

Requirements:
    pip install beautifulsoup4 tqdm requests
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

# ── Dependency check ───────────────────────────────────────────────────────────
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

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_URL = "https://dictionary.cambridge.org/dictionary/english/{word}"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
DELAY_PER_WORKER = 0.5   # seconds between requests per thread
MAX_RETRIES = 3

# Global states
abort_crawl = False
consecutive_errors = 0
state_lock = Lock()

# Queue for database writing (Producer-Consumer)
write_queue = queue.Queue()

# Thread-local storage for requests session to keep Keep-Alive active
thread_local = threading.local()


def get_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
        thread_local.session.headers.update(HEADERS)
    return thread_local.session


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

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
CREATE INDEX IF NOT EXISTS idx_synonyms_sense ON sense_synonyms(sense_id);
CREATE INDEX IF NOT EXISTS idx_collocations_word ON collocations(word_id);
"""


def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA cache_size=-10000;")
    conn.executescript(SCHEMA)
    # Migrate existing DBs missing new columns
    for col_sql, label in [
        ("ALTER TABLE words ADD COLUMN entry_type   TEXT DEFAULT 'word'", "entry_type"),
        ("ALTER TABLE words ADD COLUMN display_form TEXT",                "display_form"),
        ("ALTER TABLE entries ADD COLUMN dictionary_source TEXT",          "dictionary_source"),
        ("ALTER TABLE senses ADD COLUMN phrase_title TEXT",               "phrase_title"),
    ]:
        try:
            conn.execute(col_sql)
            conn.commit()
            print(f"  [migrate] Added {label} column")
        except Exception:
            pass  # Column already exists
            
    # Create new tables if they don't exist yet (safe to re-run)
    conn.executescript("""
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
    """)
    conn.commit()
    return conn


def load_words(conn: sqlite3.Connection, words_file: str):
    """
    Insert/update words table from a word list file. Safe to re-run.
    """
    rows: list[tuple[str, str, str]] = []   # (slug, display_form, entry_type)
    with open(words_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                rows.append((parts[0], parts[1], parts[2]))      # 3-col TSV
            elif len(parts) == 2:
                rows.append((parts[0], parts[0], parts[1]))      # 2-col TSV, display=slug
            else:
                rows.append((parts[0], parts[0], "word"))        # plain text

    cur = conn.cursor()
    # Insert new entries (skip duplicates)
    cur.executemany(
        "INSERT OR IGNORE INTO words (word, display_form, entry_type) VALUES (?, ?, ?)",
        rows,
    )
    # Update display_form + entry_type for existing rows (backfill)
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


# ═══════════════════════════════════════════════════════════════════════════════
# PRODUCER-CONSUMER DATABASE WRITER THREAD
# ═══════════════════════════════════════════════════════════════════════════════

class DbWriterThread(threading.Thread):
    """
    Dedicated database writer thread. Reads finished crawl tasks from the queue
    and commits them in batches. This completely avoids lock contention.
    """
    def __init__(self, db_path: str, batch_size: int = 50):
        super().__init__()
        self.db_path = db_path
        self.batch_size = batch_size
        self.daemon = True
        self.running = True

    def run(self):
        conn = init_db(self.db_path)
        batch = []

        while self.running or not write_queue.empty():
            try:
                # Wait for item with timeout to commit partial batches on idle
                item = write_queue.get(timeout=0.5)
                if item is None:
                    write_queue.task_done()
                    break
                batch.append(item)
                write_queue.task_done()
            except queue.Empty:
                pass

            # Flush batch if full, or if the queue is empty and we have pending writes
            if len(batch) >= self.batch_size or (len(batch) > 0 and write_queue.empty()):
                self.write_batch(conn, batch)
                batch = []

        if batch:
            self.write_batch(conn, batch)
        conn.close()

    def write_batch(self, conn: sqlite3.Connection, batch: list[tuple]):
        cur = conn.cursor()
        now = datetime.utcnow().isoformat()
        try:
            for word_id, entries_data, status, error_msg, collocations in batch:
                # Update status
                cur.execute(
                    "UPDATE words SET status=?, crawled_at=?, error_msg=? WHERE id=?",
                    (status, now, error_msg, word_id),
                )

                # Delete old
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

                    for sense_data in entry_data["senses"]:
                        cur.execute(
                            """INSERT INTO senses
                               (entry_id, sense_order, guideword, definition,
                                cefr_level, grammar, domain, labels, phrase_title)
                               VALUES (?,?,?,?,?,?,?,?,?)""",
                            (
                                entry_id,
                                sense_data["sense_order"],
                                sense_data["guideword"],
                                sense_data["definition"],
                                sense_data["cefr_level"],
                                sense_data["grammar"],
                                sense_data["domain"],
                                json.dumps(sense_data["labels"], ensure_ascii=False),
                                sense_data.get("phrase_title"),
                            ),
                        )
                        sense_id = cur.lastrowid

                        # Synonyms
                        syns = sense_data.get("synonyms", [])
                        if syns:
                            cur.executemany(
                                """INSERT INTO sense_synonyms (sense_id, synonym, slug, is_antonym)
                                   VALUES (?,?,?,?)""",
                                [(sense_id, syn["synonym"], syn["slug"], syn["is_antonym"]) for syn in syns],
                            )

                        # Examples
                        exs = sense_data["examples"]
                        if exs:
                            cur.executemany(
                                """INSERT INTO examples
                                   (sense_id, example_order, example, collocation, is_extra)
                                   VALUES (?,?,?,?,?)""",
                                [(sense_id, ex_i, ex_data["text"], ex_data["collocation"], ex_data.get("is_extra", 0))
                                 for ex_i, ex_data in enumerate(exs)],
                            )

                # Collocations
                if collocations:
                    cur.executemany(
                        """INSERT INTO collocations (word_id, collocation, example, source)
                           VALUES (?,?,?,?)""",
                        [(word_id, c["collocation"], c["example"], c["source"]) for c in collocations],
                    )

                # ── Topics & More Meanings Discovery ──────────────────────────
                all_rel_slugs = []
                topic_links = []  # list of (rel_slug, topic_id)

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

                # 1. Bulk insert all new words
                all_new_words = list(set(all_rel_slugs + more_meanings))
                if all_new_words:
                    cur.executemany(
                        "INSERT OR IGNORE INTO words (word, status) VALUES (?, 'seen')",
                        [(w,) for w in all_new_words]
                    )

                # 2. Bulk resolve word IDs and insert link mapping
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
        except Exception as e:
            conn.rollback()
            tqdm.write(f"\n[ERROR] Database write batch failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# FETCHER (Reuses Keep-Alive HTTP connection per worker thread)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch(url: str) -> tuple[str, int]:
    """Return (html, status_code). status_code=0 on network error."""
    global abort_crawl
    if abort_crawl:
        return "", 0

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
        except requests.exceptions.RequestException:
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return "", 0


# ═══════════════════════════════════════════════════════════════════════════════
# PARSER
# ═══════════════════════════════════════════════════════════════════════════════

def _text(tag) -> str:
    """Get clean text from a BS4 tag."""
    if tag is None:
        return ""
    return tag.get_text(" ", strip=True)


def _get_audio_url(block, region: str) -> str:
    """Extract MP3 audio URL for 'uk' or 'us'."""
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
    """
    Parse a Cambridge dictionary word page.
    """
    soup = BeautifulSoup(html, "html.parser")
    entries = []

    # 1. Identify entry blocks.
    # On standard pages, we look for ".entry-body__el".
    # On phrase/idiom pages, we fall back to ".di-body".
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
            "senses": [],
        }

        # ── Headword ──────────────────────────────────────────────────────────
        hw = block.select_one(".hw, .dhw, .headword")
        entry["headword"] = _text(hw)

        # ── Part of speech ────────────────────────────────────────────────────
        pos_tag = block.select_one(".pos-header .pos.dpos, .di-info .pos.dpos, .pos.dpos")
        entry["pos"] = _text(pos_tag)

        # ── Grammar code ──────────────────────────────────────────────────────
        gram_tag = block.select_one(".pos-header .posgram .gram.dgram, .di-info .gram.dgram, .posgram .gram.dgram")
        if gram_tag:
            gc = gram_tag.select(".gc.dgc")
            if gc:
                entry["grammar"] = "[ " + " ".join(_text(g) for g in gc) + " ]"
            else:
                raw = _text(gram_tag)
                entry["grammar"] = raw

        # ── Pronunciation (IPA) ───────────────────────────────────────────────
        uk_ipa = block.select_one(".pos-header .uk.dpron-i .ipa.dipa, .uk.dpron-i .ipa.dipa, .dpron-i.uk .ipa")
        entry["pronunciation_uk"] = _text(uk_ipa)

        us_ipa = block.select_one(".pos-header .us.dpron-i .ipa.dipa, .us.dpron-i .ipa.dipa, .dpron-i.us .ipa")
        entry["pronunciation_us"] = _text(us_ipa)

        # ── Audio URLs ────────────────────────────────────────────────────────
        entry["audio_uk_url"] = _get_audio_url(block, "uk")
        entry["audio_us_url"] = _get_audio_url(block, "us")

        # ── Dictionary Source ─────────────────────────────────────────────────
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

        # ── Senses ────────────────────────────────────────────────────────────
        sense_order = 0
        sense_blocks = block.select(".dsense")

        if not sense_blocks:
            # Phrase pages do not have .dsense wrappers, parse def-blocks directly
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

                # Definition text
                def_tag = def_block.select_one(".def.ddef_d")
                if def_tag:
                    sense["definition"] = _text(def_tag).rstrip(":").strip()

                if not sense["definition"]:
                    continue

                # CEFR level
                cefr_tag = def_block.select_one(".epp-xref")
                sense["cefr_level"] = _text(cefr_tag)

                # Grammar
                gram_tag = def_block.select_one(".ddef_h .gram.dgram")
                if gram_tag:
                    gc = gram_tag.select(".gc.dgc")
                    sense["grammar"] = (
                        "[ " + " ".join(_text(g) for g in gc) + " ]"
                        if gc else _text(gram_tag)
                    )

                # Domain label
                domain_tag = def_block.select_one(".domain.ddomain")
                sense["domain"] = _text(domain_tag)

                # Labels
                lab_tags = def_block.select(".lab.dlab .usage.dusage")
                sense["labels"] = [_text(l) for l in lab_tags if _text(l)]

                # Phrase parent
                phrase_parent = def_block.find_parent(class_="phrase-block") or def_block.find_parent(class_="idiom-block")
                if phrase_parent:
                    phrase_title_tag = phrase_parent.select_one(".phrase-title, .di-title")
                    if phrase_title_tag:
                        sense["phrase_title"] = phrase_title_tag.text.strip()

                # Synonyms inside ddef_block
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

                # Examples
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
            # Standard .dsense wrapper block loop
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

                    # Definition text
                    def_tag = def_block.select_one(".def.ddef_d")
                    if def_tag:
                        sense["definition"] = _text(def_tag).rstrip(":").strip()

                    if not sense["definition"]:
                        continue

                    # CEFR level
                    cefr_tag = def_block.select_one(".epp-xref")
                    sense["cefr_level"] = _text(cefr_tag)

                    # Grammar
                    gram_tag = def_block.select_one(".ddef_h .gram.dgram")
                    if gram_tag:
                        gc = gram_tag.select(".gc.dgc")
                        sense["grammar"] = (
                            "[ " + " ".join(_text(g) for g in gc) + " ]"
                            if gc else _text(gram_tag)
                        )

                    # Domain label
                    domain_tag = def_block.select_one(".domain.ddomain")
                    sense["domain"] = _text(domain_tag)

                    # Labels
                    lab_tags = def_block.select(".lab.dlab .usage.dusage")
                    sense["labels"] = [_text(l) for l in lab_tags if _text(l)]

                    # Phrase parent
                    phrase_parent = def_block.find_parent(class_="phrase-block") or def_block.find_parent(class_="idiom-block")
                    if phrase_parent:
                        phrase_title_tag = phrase_parent.select_one(".phrase-title, .di-title")
                        if phrase_title_tag:
                            sense["phrase_title"] = phrase_title_tag.text.strip()

                    # Synonyms inside ddef_block
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

                    # Examples
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

                    # Extra examples inside dsense
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

    # ── SMART Vocabulary topics ───────────────────────────────────────────────
    topics: list[dict] = []
    for smart_block in soup.select(".smartt.daccord"):
        topic_link = smart_block.select_one(".daccord_lt a")
        if not topic_link:
            continue
        topic_url = topic_link.get("href", "").strip()
        topic_title = _text(topic_link)
        import re
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

    # ── More Meanings Sidebar ─────────────────────────────────────────────────
    more_meanings_slugs = []
    for container in soup.select(".cdo-more-results"):
        for a in container.select("a[href*='/dictionary/english/']"):
            href = a.get("href", "")
            import re
            m = re.search(r'/dictionary/english/([^?\s]+)', href)
            if m:
                more_meanings_slugs.append(m.group(1).strip())

    # Check if there is any link to collocation on the page
    has_collocations = False
    if soup.select_one("a[href*='/collocation/']"):
        has_collocations = True

    if entries:
        entries[0]["topics"] = topics
        entries[0]["more_meanings"] = more_meanings_slugs
        entries[0]["has_collocations"] = has_collocations

    return entries


def parse_collocation_page(html: str) -> list[dict]:
    """
    Parse a Cambridge collocation page.
    """
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


# ═══════════════════════════════════════════════════════════════════════════════
# WORKER (Push results to queue immediately)
# ═══════════════════════════════════════════════════════════════════════════════

def process_word(
    word_id: int,
    word: str,
    entry_type: str,
) -> str:
    """Fetch, parse, and queue one word. Returns status string."""
    global abort_crawl, consecutive_errors
    with state_lock:
        if abort_crawl:
            return "error"

    time.sleep(DELAY_PER_WORKER)
    url = BASE_URL.format(word=word)
    html, code = fetch(url)

    if code == 404 or (code == 200 and not html):
        write_queue.put((word_id, [], "not_found", None, None))
        with state_lock:
            consecutive_errors = 0
        return "not_found"

    if code != 200:
        write_queue.put((word_id, [], "error", f"HTTP Error status: {code}", None))
        with state_lock:
            consecutive_errors += 1
            if consecutive_errors >= 5:
                abort_crawl = True
                tqdm.write(f"\n[CRITICAL] 5 consecutive network errors! Last HTTP status: {code}. Aborting...")
        return "error"

    try:
        entries = parse_page(html)
    except Exception as e:
        write_queue.put((word_id, [], "error", f"Parse error: {e}", None))
        return "error"

    if not entries:
        write_queue.put((word_id, [], "not_found", "No entries found", None))
        with state_lock:
            consecutive_errors = 0
        return "not_found"

    # Fetch collocations (Item 6) - only if the main page explicitly links to one
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

    # Push to queue to write in bulk asynchronously
    write_queue.put((word_id, entries, "done", None, collocations))
    
    with state_lock:
        consecutive_errors = 0

    return "done"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

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


def main():
    global abort_crawl
    parser = argparse.ArgumentParser(
        description="Cambridge Dictionary crawler → SQLite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First run - load words and start crawling
  python crawl_dictionary.py --words cambridge_words.txt --db cambridge.db

  # Just run again to resume (automatically picks up where it left off)
  python crawl_dictionary.py --db cambridge.db

  # Use more workers (be careful with rate limits)
  python crawl_dictionary.py --words cambridge_words.txt --db cambridge.db --workers 8

  # Load words only, don't crawl yet
  python crawl_dictionary.py --words cambridge_words.txt --db cambridge.db --load-only
        """,
    )
    parser.add_argument("--words", help="Path to word list file (one word per line). Optional if DB already has words.")
    parser.add_argument("--db", default="cambridge.db", help="SQLite database path (default: cambridge.db)")
    parser.add_argument("--workers", type=int, default=5, help="Number of concurrent workers (default: 5)")
    parser.add_argument("--load-only", action="store_true", help="Only load words into DB, don't crawl")
    parser.add_argument("--stats", action="store_true", help="Show DB stats and exit")
    args = parser.parse_args()

    print(f"Cambridge Dictionary Crawler")
    print(f"  DB       : {args.db}")
    print(f"  Workers  : {args.workers}")

    # Load and resume DB connections
    conn = init_db(args.db)

    if args.stats:
        print_stats(conn)
        conn.close()
        return

    # Load words file if provided
    if args.words:
        print(f"\n[1/2] Loading words from: {args.words}")
        load_words(conn, args.words)

    if args.load_only:
        print_stats(conn)
        conn.close()
        return

    # Get pending words
    pending = get_pending(conn)
    conn.close() # Close connection so DB Writer Thread can have exclusive write access

    if not pending:
        conn = init_db(args.db)
        print("\nNo pending words to crawl.")
        print_stats(conn)
        conn.close()
        return

    print(f"\n[2/2] Crawling {len(pending)} pending words with {args.workers} workers...")
    print(f"       Rate: ~{DELAY_PER_WORKER}s delay/worker → ~{DELAY_PER_WORKER/args.workers:.2f}s avg between requests")
    print()

    # Start the DB writer thread
    writer = DbWriterThread(args.db, batch_size=50)
    writer.start()

    counters = {"done": 0, "error": 0, "not_found": 0}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_word, wid, word, etype): word
            for wid, word, etype in pending
        }

        with tqdm(
            total=len(pending),
            unit="word",
            dynamic_ncols=True,
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
                    nf=counters["not_found"],
                    refresh=False,
                )
                pbar.update(1)

    # Stop the DB writer thread and flush remaining writes
    writer.running = False
    write_queue.put(None) # poison pill to exit thread
    writer.join()

    # Reopen DB connection for printing final statistics
    conn = init_db(args.db)
    print()
    if abort_crawl:
        print("[ABORTED] Crawler was stopped early due to consecutive network errors (suspected IP block/ban).\n")
    print_stats(conn)
    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
