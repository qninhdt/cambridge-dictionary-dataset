#!/usr/bin/env python3
"""
Cambridge Dictionary Browse Scraper (Full)
==========================================
Scrapes ALL entries from Cambridge browse pages including:
  - Single words       (word)
  - Phrasal verbs      (phrasal_verb)
  - Idioms             (idiom)
  - Phrases            (phrase)
  - Compound/others    (expression)

Output: cambridge_entries.tsv   (slug TAB entry_type)
"""

import re
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

BASE = "https://dictionary.cambridge.org"
OUTPUT_FILE = "cambridge_entries.tsv"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}
MAX_WORKERS = 5
DELAY = 0.3

print_lock = Lock()
abort_scrape = False
consecutive_errors = 0


def fetch(url: str, retries: int = 3) -> str:
    global abort_scrape, consecutive_errors
    with print_lock:
        if abort_scrape:
            return ""

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=15) as r:
                with print_lock:
                    consecutive_errors = 0
                return r.read().decode("utf-8", errors="ignore")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                with print_lock:
                    print(f"\n[WARNING] Rate limited (429) on {url}. Retrying in {10 * (attempt + 1)}s...")
                time.sleep(10 * (attempt + 1))
            elif e.code == 403:
                with print_lock:
                    print(f"\n[WARNING] Forbidden (403) on {url}. Suspected IP block.")
                time.sleep(5 * (attempt + 1))
            elif e.code == 404:
                return ""
            else:
                time.sleep(2)
        except Exception:
            time.sleep(2)

    with print_lock:
        consecutive_errors += 1
        if consecutive_errors >= 5:
            abort_scrape = True
            print("\n[CRITICAL] 5 consecutive network errors! Suspected IP ban or network down. Aborting scrape...")
    return ""


# Suffixes Cambridge appends to the display word in title attributes
_TYPE_SUFFIXES = [
    ("phrasal verb", "phrasal_verb"),
    ("idiom",        "idiom"),
    ("phrase",       "phrase"),
]
_TITLE_STRIP = re.compile(
    r"\s*(phrasal verb|idiom|phrase|definition).*$", re.IGNORECASE
)


def parse_title(slug: str, title: str) -> tuple[str, str]:
    """
    Return (display_form, entry_type) from a Cambridge browse link.

    title examples:
      "give up phrasal verb definition in English"  → ("give up",           "phrasal_verb")
      "kick the bucket idiom definition in English" → ("kick the bucket",   "idiom")
      "well-being definition in English"            → ("well-being",        "expression")
      "abandon definition in English"               → ("abandon",           "word")
    """
    t_lower = title.lower()

    # Determine entry_type from explicit label
    entry_type = None
    for label, etype in _TYPE_SUFFIXES:
        if label in t_lower:
            entry_type = etype
            break
    if entry_type is None:
        entry_type = "word" if ("-" not in slug and slug.replace("'", "").isalpha()) else "expression"

    # Extract display form: strip trailing type label + "definition in English"
    display = _TITLE_STRIP.sub("", title).strip()
    if not display:
        # fallback: replace hyphens with spaces (might be wrong for compounds)
        display = slug.replace("-", " ")

    return display, entry_type


# ── Level 1: letter pages ─────────────────────────────────────────────────────
def get_letter_urls() -> list[str]:
    html = fetch(f"{BASE}/browse/english/")
    links = re.findall(
        r'href="(https://dictionary\.cambridge\.org/browse/english/[^/\"]+/)"', html
    )
    links += [BASE + x for x in re.findall(r'href="(/browse/english/[^/\"]+/)"', html)]
    seen, result = set(), []
    for l in links:
        if l not in seen:
            seen.add(l)
            result.append(l)
    print(f"[L1] {len(result)} letter pages")
    return result


# ── Level 2: chunk pages per letter ──────────────────────────────────────────
def get_chunk_urls(letter_url: str) -> list[str]:
    time.sleep(DELAY)
    html = fetch(letter_url)
    links = re.findall(
        r'href="(https://dictionary\.cambridge\.org/browse/english/[^/]+/[^/\"]+/)"', html
    )
    links += [BASE + x for x in re.findall(r'href="(/browse/english/[^/]+/[^/\"]+/)"', html)]
    seen, result = set(), []
    for l in links:
        parts = l.rstrip("/").split("/")
        if len(parts) >= 7 and l not in seen:
            seen.add(l)
            result.append(l)
    return result


# ── Level 3: entries from a chunk page ───────────────────────────────────────
def get_entries_from_chunk(chunk_url: str) -> list[tuple[str, str, str]]:
    """Returns list of (slug, display_form, entry_type)."""
    time.sleep(DELAY)
    html = fetch(chunk_url)

    # Extract slug + title from all /dictionary/english/{slug} links
    pattern = re.compile(r'href="/dictionary/english/([^"]+)"[^>]*title="([^"]*)"')
    seen: set[str] = set()
    result: list[tuple[str, str, str]] = []

    for m in pattern.finditer(html):
        slug = m.group(1).strip()
        title = m.group(2).strip()
        if slug and slug not in seen:
            seen.add(slug)
            display, etype = parse_title(slug, title)
            result.append((slug, display, etype))

    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import time as _time
    start = _time.time()

    # Step 1 – letter pages
    letter_urls = get_letter_urls()
    if not letter_urls:
        print("ERROR: Could not fetch letter pages.")
        return

    # Step 2 – chunk pages
    print("[L2] Collecting chunk pages...")
    all_chunks: list[str] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(get_chunk_urls, u): u for u in letter_urls}
        for f in as_completed(futures):
            chunks = f.result()
            all_chunks.extend(chunks)
    print(f"[L2] {len(all_chunks)} chunk pages")

    # Step 3 – entries from chunks
    print(f"[L3] Collecting entries...")
    all_entries: dict[str, tuple[str, str]] = {}   # slug → (display_form, entry_type)
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(get_entries_from_chunk, u): u for u in all_chunks}
        for f in as_completed(futures):
            entries = f.result()
            for slug, display, etype in entries:
                all_entries[slug] = (display, etype)
            done += 1
            if done % 200 == 0:
                elapsed = _time.time() - start
                eta = (len(all_chunks) - done) * elapsed / done
                with print_lock:
                    print(f"  [{done}/{len(all_chunks)}] entries={len(all_entries)} "
                          f"elapsed={elapsed:.0f}s ETA={eta:.0f}s")

    # Save TSV  (3 columns: slug TAB display_form TAB entry_type)
    counts: dict[str, int] = {}
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for slug, (display, etype) in sorted(all_entries.items()):
            f.write(f"{slug}\t{display}\t{etype}\n")
            counts[etype] = counts.get(etype, 0) + 1

    elapsed = _time.time() - start
    if abort_scrape:
        print("\n[ABORTED] Scrape was stopped early due to consecutive network errors (suspected IP block/ban).")
    print(f"\nDone in {elapsed:.0f}s → {OUTPUT_FILE}")
    print(f"  Total        : {len(all_entries)}")
    for etype, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {etype:<16}: {n}")


if __name__ == "__main__":
    main()
