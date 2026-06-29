import gzip
import sqlite3
import re
import os
import json
from tqdm import tqdm

def parse_wordnet_url(url):
    # E.g. http://wordnet-rdf.princeton.edu/wn31/302193771-a/ -> oewn-02193771-a
    # E.g. http://wordnet-rdf.princeton.edu/wn30/302193771-a  -> oewn-02193771-a
    match = re.search(r'/wn3[01]/(\d)-?(\d{8})-?([nvasr])', url)
    if match:
        pos_digit, offset, pos_char = match.groups()
        # WordNet satellite adjectives 's' are often represented as 'a' in ConceptNet/PWN31
        return f"oewn-{offset}-{pos_char}"
    return None

def main():
    db_path = "data/dictionary_alignment.db"
    conceptnet_path = "conceptnet-assertions-5.7.0.csv.gz"

    if not os.path.exists(conceptnet_path):
        print(f"Error: {conceptnet_path} not found in the root directory.")
        return

    print("Connecting to database...")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Create tables
    c.execute("""
        CREATE TABLE IF NOT EXISTS conceptnet_sense_mappings (
            conceptnet_uri TEXT PRIMARY KEY,
            wordnet_id TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS conceptnet_relations (
            start_concept TEXT,
            relation TEXT,
            end_concept TEXT,
            weight REAL
        )
    """)
    
    # Clear existing to ensure deterministic run
    c.execute("DELETE FROM conceptnet_sense_mappings")
    c.execute("DELETE FROM conceptnet_relations")
    conn.commit()

    # Refened relations we want to import
    target_relations = {
        '/r/UsedFor', '/r/CapableOf', '/r/HasProperty', 
        '/r/PartOf', '/r/MadeOf', '/r/Causes', 
        '/r/Synonym', '/r/Antonym'
    }

    print("Parsing ConceptNet assertions...")
    
    mappings_batch = []
    relations_batch = []
    
    batch_size = 50000
    pbar = tqdm(desc="Processing ConceptNet lines", unit=" lines")

    with gzip.open(conceptnet_path, 'rt', encoding='utf-8') as f:
        for line in f:
            pbar.update(1)
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            
            uri, rel, start, end, metadata_json = parts[:5]
            
            # 1. Extract WordNet ExternalURL mappings
            if rel == '/r/ExternalURL':
                if start.startswith('/c/en/') and start.endswith('/wn') and 'wordnet-rdf.princeton.edu' in end:
                    wn_id = parse_wordnet_url(end)
                    if wn_id:
                        mappings_batch.append((start, wn_id))

            # 2. Extract refined English relations
            elif rel in target_relations:
                if start.startswith('/c/en/') and end.startswith('/c/en/'):
                    # Skip self-loops
                    if start == end:
                        continue
                    
                    try:
                        meta = json.loads(metadata_json)
                        weight = float(meta.get('weight', 1.0))
                    except Exception:
                        weight = 1.0
                        
                    relations_batch.append((start, rel, end, weight))

            # Commit batches
            if len(mappings_batch) >= batch_size:
                c.executemany("INSERT OR REPLACE INTO conceptnet_sense_mappings VALUES (?, ?)", mappings_batch)
                mappings_batch = []
                conn.commit()

            if len(relations_batch) >= batch_size:
                c.executemany("INSERT INTO conceptnet_relations VALUES (?, ?, ?, ?)", relations_batch)
                relations_batch = []
                conn.commit()

    # Insert remaining
    if mappings_batch:
        c.executemany("INSERT OR REPLACE INTO conceptnet_sense_mappings VALUES (?, ?)", mappings_batch)
    if relations_batch:
        c.executemany("INSERT INTO conceptnet_relations VALUES (?, ?, ?, ?)", relations_batch)
    
    conn.commit()
    pbar.close()

    # Create indexes for performance
    print("Creating indexes...")
    c.execute("CREATE INDEX IF NOT EXISTS idx_conceptnet_relations_start ON conceptnet_relations(start_concept)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_conceptnet_sense_mappings_wn ON conceptnet_sense_mappings(wordnet_id)")
    conn.commit()

    # Verify counts
    c.execute("SELECT count(*) FROM conceptnet_sense_mappings")
    mappings_count = c.fetchone()[0]
    c.execute("SELECT count(*) FROM conceptnet_relations")
    relations_count = c.fetchone()[0]

    print("\n=== IMPORT COMPLETE ===")
    print(f"Total Sense Mappings: {mappings_count:,}")
    print(f"Total Relations:      {relations_count:,}")

    conn.close()

if __name__ == "__main__":
    main()
