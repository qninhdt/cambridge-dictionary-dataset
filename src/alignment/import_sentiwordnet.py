import sqlite3
import os
import wn
import nltk
from nltk.corpus import wordnet as wn_nltk
from nltk.corpus import sentiwordnet as swn
from tqdm import tqdm

def main():
    db_path = "data/dictionary_alignment.db"

    if not os.path.exists(db_path):
        print(f"Error: Database {db_path} not found.")
        return

    # Ensure NLTK corpora are downloaded
    print("Verifying NLTK corpora...")
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")
    try:
        nltk.data.find("corpora/sentiwordnet")
    except LookupError:
        nltk.download("sentiwordnet")

    print("Loading Open English WordNet 2024...")
    oewn = wn.Wordnet('oewn:2024')

    print("Connecting to database...")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Create table
    c.execute("""
        CREATE TABLE IF NOT EXISTS wordnet_sentiment (
            wordnet_id TEXT PRIMARY KEY,
            pos_score REAL,
            neg_score REAL,
            obj_score REAL
        )
    """)
    
    # Clear existing to ensure deterministic run
    c.execute("DELETE FROM wordnet_sentiment")
    conn.commit()

    # Fetch all unique WordNet IDs currently in our links
    c.execute("SELECT DISTINCT wordnet_id FROM unified_sense_wordnet_links WHERE wordnet_id LIKE 'oewn-%'")
    db_ids = [r[0] for r in c.fetchall()]
    
    print(f"Total unique WordNet IDs to process: {len(db_ids):,}")

    sentiment_batch = []
    batch_size = 1000
    
    mapped_count = 0
    default_count = 0

    for db_id in tqdm(db_ids, desc="Importing SentiWordNet scores"):
        pos_score = 0.0
        neg_score = 0.0
        obj_score = 1.0
        is_mapped = False

        try:
            # 1. Lookup synset in OEWN 2024
            ss = oewn.synset(db_id)
            senses = ss.senses()
            if senses:
                # 2. Get first sense and translate to Sense Key
                sense_id = senses[0].id
                core = sense_id[5:].replace('__', '%').replace('.', ':')
                
                # 3. Lookup in NLTK WordNet and SentiWordNet
                lemma_nltk = wn_nltk.lemma_from_key(core)
                if lemma_nltk:
                    synset_nltk = lemma_nltk.synset()
                    senti_ss = swn.senti_synset(synset_nltk.name())
                    if senti_ss:
                        pos_score = senti_ss.pos_score()
                        neg_score = senti_ss.neg_score()
                        obj_score = senti_ss.obj_score()
                        is_mapped = True
        except Exception:
            # Any failure (e.g. new OEWN synset not in PWN 3.0) will fallback to default objective scores
            pass

        if is_mapped:
            mapped_count += 1
        else:
            default_count += 1

        sentiment_batch.append((db_id, pos_score, neg_score, obj_score))

        if len(sentiment_batch) >= batch_size:
            c.executemany("INSERT OR REPLACE INTO wordnet_sentiment VALUES (?, ?, ?, ?)", sentiment_batch)
            sentiment_batch = []
            conn.commit()

    # Insert remaining
    if sentiment_batch:
        c.executemany("INSERT OR REPLACE INTO wordnet_sentiment VALUES (?, ?, ?, ?)", sentiment_batch)
        conn.commit()

    # Verify counts
    c.execute("SELECT count(*) FROM wordnet_sentiment")
    imported_count = c.fetchone()[0]

    print("\n=== SENTIMENT IMPORT COMPLETE ===")
    print(f"Total Synsets Processed:  {imported_count:,}")
    print(f"  - Successfully Mapped:  {mapped_count:,} ({mapped_count/imported_count*100:.2f}%)")
    print(f"  - Defaulted (Neutral):  {default_count:,} ({default_count/imported_count*100:.2f}%)")

    conn.close()

if __name__ == "__main__":
    main()
