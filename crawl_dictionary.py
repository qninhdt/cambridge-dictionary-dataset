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
    pip install beautifulsoup4 tqdm
"""

import argparse
import json
import re
import sqlite3
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock

# ── Dependency check ───────────────────────────────────────────────────────────
try:
    from bs4 import BeautifulSoup
except ImportError:
    print("Missing dependency: pip install beautifulsoup4 tqdm")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Missing dependency: pip install beautifulsoup4 tqdm")
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

# Global abort states
abort_crawl = False
consecutive_errors = 0


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

    Supported formats:
      - 1-column plain text:  slug
      - 2-column TSV:         slug  TAB  entry_type          (old format)
      - 3-column TSV:         slug  TAB  display_form  TAB  entry_type  (new format)
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


def get_pending(conn: sqlite3.Connection) -> list[tuple[int, str]]:
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT id, word FROM words WHERE status='pending' ORDER BY id"
    ).fetchall()
    return rows


def save_result(
    conn: sqlite3.Connection,
    lock: Lock,
    word_id: int,
    entries_data: list[dict],
    status: str,
    error_msg: str = None,
    collocations: list[dict] = None,
):
    with lock:
        cur = conn.cursor()
        now = datetime.utcnow().isoformat()

        # Update word status
        cur.execute(
            "UPDATE words SET status=?, crawled_at=?, error_msg=? WHERE id=?",
            (status, now, error_msg, word_id),
        )

        # Delete old data (re-crawl support)
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

                # Save synonyms in bulk
                syns = sense_data.get("synonyms", [])
                if syns:
                    cur.executemany(
                        """INSERT INTO sense_synonyms (sense_id, synonym, slug, is_antonym)
                           VALUES (?,?,?,?)""",
                        [(sense_id, syn["synonym"], syn["slug"], syn["is_antonym"]) for syn in syns],
                    )

                # Save examples in bulk
                exs = sense_data["examples"]
                if exs:
                    cur.executemany(
                        """INSERT INTO examples
                           (sense_id, example_order, example, collocation, is_extra)
                           VALUES (?,?,?,?,?)""",
                        [(sense_id, ex_i, ex_data["text"], ex_data["collocation"], ex_data.get("is_extra", 0))
                         for ex_i, ex_data in enumerate(exs)],
                    )

        # Save collocations in bulk
        if collocations:
            cur.executemany(
                """INSERT INTO collocations (word_id, collocation, example, source)
                   VALUES (?,?,?,?)""",
                [(word_id, c["collocation"], c["example"], c["source"]) for c in collocations],
            )

        # ── Topics & More Meanings Discovery (Optimized Bulk Insert) ──────────
        all_rel_slugs = []
        topic_links = []  # list of (rel_slug, topic_id)

        for topic_data in entries_data[0].get("topics", []) if entries_data else []:
            # Insert topic globally
            cur.execute(
                "INSERT OR IGNORE INTO topics (slug, title, url) VALUES (?,?,?)",
                (topic_data["slug"], topic_data["title"], topic_data["url"]),
            )
            cur.execute("SELECT id FROM topics WHERE slug=?", (topic_data["slug"],))
            topic_id = cur.fetchone()[0]

            # Link current word to this topic
            cur.execute(
                "INSERT OR IGNORE INTO word_topics (word_id, topic_id) VALUES (?,?)",
                (word_id, topic_id),
            )

            # Collect related slugs to batch later
            for rel_slug in topic_data.get("related_slugs", []):
                all_rel_slugs.append(rel_slug)
                topic_links.append((rel_slug, topic_id))

        more_meanings = entries_data[0].get("more_meanings", []) if entries_data else []

        # 1. Bulk insert all new words (related + more meanings) in one query
        all_new_words = list(set(all_rel_slugs + more_meanings))
        if all_new_words:
            cur.executemany(
                "INSERT OR IGNORE INTO words (word, status) VALUES (?, 'seen')",
                [(w,) for w in all_new_words]
            )

        # 2. Bulk resolve word IDs for related words and insert link mapping
        if topic_links:
            unique_rel_slugs = list(set(all_rel_slugs))
            slug_to_id = {}
            # Chunk to avoid SQLite parameter limit (999 parameters max)
            for i in range(0, len(unique_rel_slugs), 900):
                chunk = unique_rel_slugs[i : i + 900]
                placeholders = ",".join("?" for _ in chunk)
                cur.execute(
                    f"SELECT word, id FROM words WHERE word IN ({placeholders})",
                    chunk,
                )
                for w, wid in cur.fetchall():
                    slug_to_id[w] = wid

            # Prepare bulk insert into word_topics
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


# ═══════════════════════════════════════════════════════════════════════════════
# FETCHER
# ═══════════════════════════════════════════════════════════════════════════════

def fetch(url: str) -> tuple[str, int]:
    """Return (html, status_code). status_code=0 on network error."""
    global abort_crawl
    if abort_crawl:
        return "", 0

    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=20) as resp:
                return resp.read().decode("utf-8", errors="ignore"), resp.status
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return "", 404
            if e.code == 429:
                tqdm.write(f"\n[WARNING] Rate limited (429) on {url}. Retrying in {10 * (attempt + 1)}s...")
                time.sleep(10 * (attempt + 1))
            elif e.code == 403:
                tqdm.write(f"\n[WARNING] Forbidden (403) on {url}. Possible IP block.")
                time.sleep(5 * (attempt + 1))
            else:
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
    Returns a list of entry dicts, each containing:
      entry_order, headword, pos, grammar,
      pronunciation_uk, pronunciation_us, audio_uk_url, audio_us_url,
      dictionary_source,
      topics: [{slug, title, url, related_slugs}]  (on first entry only)
      more_meanings: [slug]                        (on first entry only)
      senses: [{sense_order, guideword, definition, cefr_level,
                grammar, domain, labels, phrase_title,
                synonyms: [{synonym, slug, is_antonym}],
                examples: [{text, collocation, is_extra}]}]
    """
    soup = BeautifulSoup(html, "html.parser")
    entries = []

    # Each .entry-body__el is a separate dictionary entry (verb / noun / etc.)
    entry_blocks = soup.select(".entry-body__el")
    if not entry_blocks:
        return entries

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
        hw = block.select_one(".hw.dhw")
        entry["headword"] = _text(hw)

        # ── Part of speech ────────────────────────────────────────────────────
        pos_tag = block.select_one(".pos-header .pos.dpos")
        entry["pos"] = _text(pos_tag)

        # ── Grammar code ──────────────────────────────────────────────────────
        gram_tag = block.select_one(".pos-header .posgram .gram.dgram")
        if gram_tag:
            # inner .gc tags contain the actual code letter
            gc = gram_tag.select(".gc.dgc")
            if gc:
                entry["grammar"] = "[ " + " ".join(_text(g) for g in gc) + " ]"
            else:
                raw = _text(gram_tag)
                entry["grammar"] = raw

        # ── Pronunciation (IPA) ───────────────────────────────────────────────
        uk_ipa = block.select_one(".pos-header .uk.dpron-i .ipa.dipa")
        entry["pronunciation_uk"] = _text(uk_ipa)

        us_ipa = block.select_one(".pos-header .us.dpron-i .ipa.dipa")
        entry["pronunciation_us"] = _text(us_ipa)

        # ── Audio URLs ────────────────────────────────────────────────────────
        entry["audio_uk_url"] = _get_audio_url(block, "uk")
        entry["audio_us_url"] = _get_audio_url(block, "us")

        # ── Dictionary Source (Item 3) ────────────────────────────────────────
        dict_container = block.find_parent(class_="dictionary")
        if dict_container:
            header = dict_container.select_one("h2.c_hh, .di-title")
            if header:
                src_text = header.text.strip()
                # If the header matches the headword (e.g. just "run"), normalize to British English
                if src_text.lower() == entry["headword"].lower():
                    entry["dictionary_source"] = "Cambridge Advanced Learner's Dictionary"
                else:
                    entry["dictionary_source"] = src_text

        # ── Senses ────────────────────────────────────────────────────────────
        sense_order = 0
        sense_blocks = block.select(".dsense")

        for sense_block in sense_blocks:
            # Guide word for this sense group (e.g. "LEAVE")
            gw_tag = sense_block.select_one(".guideword.dsense_gw")
            guideword = _text(gw_tag).strip("()")

            # Each .ddef_block is one definition
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
                    continue  # skip empty def blocks

                # CEFR level (B2, C1, etc.)
                cefr_tag = def_block.select_one(".epp-xref")
                sense["cefr_level"] = _text(cefr_tag)

                # Grammar (sense level)
                gram_tag = def_block.select_one(".ddef_h .gram.dgram")
                if gram_tag:
                    gc = gram_tag.select(".gc.dgc")
                    sense["grammar"] = (
                        "[ " + " ".join(_text(g) for g in gc) + " ]"
                        if gc else _text(gram_tag)
                    )

                # Domain label (COMPUTING, MEDICAL, etc.)
                domain_tag = def_block.select_one(".domain.ddomain")
                sense["domain"] = _text(domain_tag)

                # Register/usage labels (formal, informal, etc.)
                lab_tags = def_block.select(".lab.dlab .usage.dusage")
                sense["labels"] = [_text(l) for l in lab_tags if _text(l)]

                # Check if this def_block belongs to an inline phrase-block (Item 1)
                phrase_parent = def_block.find_parent(class_="phrase-block")
                if phrase_parent:
                    phrase_title_tag = phrase_parent.select_one(".phrase-title")
                    if phrase_title_tag:
                        sense["phrase_title"] = phrase_title_tag.text.strip()

                # Parse synonyms/antonyms inside this def_block (Item 2)
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

                # Standard examples inside this def block
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

            # ── "More examples" & "Thesaurus" accordions (sense-level) ─────────
            if entry["senses"]:
                last_sense = entry["senses"][-1]
                seen_texts = {ex["text"] for ex in last_sense["examples"]}
                for daccord in sense_block.select(".daccord"):
                    if "smartt" in (daccord.get("class") or []):
                        continue  # skip SMART vocab block
                        
                    # Check if it is a thesaurus block (Item 2)
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
                            
                            last_sense["synonyms"].append({
                                "synonym": a.text.strip(),
                                "slug": slug,
                                "is_antonym": is_antonym
                            })
                        continue  # skip example parsing for thesaurus accordion
                        
                    # Otherwise it's a "More examples" accordion
                    for li in daccord.select("li.eg.dexamp.hax"):
                        text = _text(li)
                        if text and text not in seen_texts:
                            last_sense["examples"].append({
                                "text": text,
                                "collocation": "",
                                "is_extra": 1,
                            })
                            seen_texts.add(text)

        # Only add entries that have real content
        if entry["headword"] or entry["pos"] or entry["senses"]:
            entries.append(entry)

    # ── SMART Vocabulary topics (at page level, attached to first entry) ──────
    topics: list[dict] = []
    for smart_block in soup.select(".smartt.daccord"):
        topic_link = smart_block.select_one(".daccord_lt a")
        if not topic_link:
            continue
        topic_url = topic_link.get("href", "").strip()
        topic_title = _text(topic_link)
        # Extract slug from URL
        slug_match = re.search(r'/topics/[^/]+/([^/?]+)/?', topic_url)
        if not slug_match:
            slug_match = re.search(r'/([^/?]+)/?$', topic_url)
        topic_slug = slug_match.group(1) if slug_match else topic_url

        # Related word slugs
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

    # ── More Meanings Sidebar (Item 4) ────────────────────────────────────────
    more_meanings_slugs = []
    for container in soup.select(".cdo-more-results"):
        for a in container.select("a[href*='/dictionary/english/']"):
            href = a.get("href", "")
            m = re.search(r'/dictionary/english/([^?\s]+)', href)
            if m:
                more_meanings_slugs.append(m.group(1).strip())

    if entries:
        entries[0]["topics"] = topics
        entries[0]["more_meanings"] = more_meanings_slugs

    return entries


def parse_collocation_page(html: str) -> list[dict]:
    """
    Parse a Cambridge collocation page (Item 6).
    Returns list of dicts: [{"collocation": str, "example": str, "source": str}]
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
# WORKER
# ═══════════════════════════════════════════════════════════════════════════════

def process_word(
    word_id: int,
    word: str,
    conn: sqlite3.Connection,
    lock: Lock,
) -> str:
    """Fetch, parse, and save one word. Returns status string."""
    global abort_crawl, consecutive_errors
    with lock:
        if abort_crawl:
            return "error"

    time.sleep(DELAY_PER_WORKER)
    url = BASE_URL.format(word=word)
    html, code = fetch(url)

    if code == 404 or (code == 200 and not html):
        save_result(conn, lock, word_id, [], "not_found")
        with lock:
            consecutive_errors = 0
        return "not_found"

    if code == 0:
        save_result(conn, lock, word_id, [], "error", "Network error after retries")
        with lock:
            consecutive_errors += 1
            if consecutive_errors >= 5:
                abort_crawl = True
                tqdm.write("\n[CRITICAL] 5 consecutive network errors! Suspected IP ban or network down. Aborting...")
        return "error"

    try:
        entries = parse_page(html)
    except Exception as e:
        save_result(conn, lock, word_id, [], "error", f"Parse error: {e}")
        return "error"

    if not entries:
        save_result(conn, lock, word_id, [], "not_found", "No entries found")
        with lock:
            consecutive_errors = 0
        return "not_found"

    # Fetch collocations (Item 6)
    time.sleep(DELAY_PER_WORKER)
    colloc_url = f"https://dictionary.cambridge.org/collocation/english/{word}"
    colloc_html, colloc_code = fetch(colloc_url)
    
    collocations = []
    if colloc_code == 200 and colloc_html:
        try:
            collocations = parse_collocation_page(colloc_html)
        except Exception:
            pass # Collocation parse failure is non-blocking

    save_result(conn, lock, word_id, entries, "done", collocations=collocations)
    
    with lock:
        consecutive_errors = 0

    return "done"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def print_stats(conn: sqlite3.Connection):
    cur = conn.cursor()
    # Status breakdown
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
    # Type breakdown
    type_rows = cur.execute(
        "SELECT entry_type, COUNT(*), SUM(status='done') FROM words GROUP BY entry_type"
    ).fetchall()
    print(f"\n── By Type ────────────────────────────")
    for etype, cnt, dcnt in sorted(type_rows, key=lambda x: -x[1]):
        print(f"  {etype:<16}: {cnt:>6}  (done: {dcnt or 0})")
    print(f"───────────────────────────────────────")


def main():
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

    conn = init_db(args.db)

    if args.stats:
        print_stats(conn)
        return

    # Load words file if provided (idempotent — safe to re-run with same file)
    if args.words:
        print(f"\n[1/2] Loading words from: {args.words}")
        load_words(conn, args.words)

    if args.load_only:
        print_stats(conn)
        return

    # Get pending words — auto-resumes from previous runs
    pending = get_pending(conn)
    if not pending:
        print("\nNo pending words to crawl.")
        print_stats(conn)
        return

    print(f"\n[2/2] Crawling {len(pending)} pending words with {args.workers} workers...")
    print(f"       Rate: ~{DELAY_PER_WORKER}s delay/worker → ~{DELAY_PER_WORKER/args.workers:.2f}s avg between requests")
    print()

    lock = Lock()
    counters = {"done": 0, "error": 0, "not_found": 0}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_word, wid, word, conn, lock): word
            for wid, word in pending
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
                except Exception as e:
                    status = "error"

                counters[status] = counters.get(status, 0) + 1
                pbar.set_postfix(
                    done=counters["done"],
                    err=counters["error"],
                    nf=counters["not_found"],
                    refresh=False,
                )
                pbar.update(1)

    print()
    if abort_crawl:
        print("[ABORTED] Crawler was stopped early due to consecutive network errors (suspected IP block/ban).\n")
    print_stats(conn)
    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
