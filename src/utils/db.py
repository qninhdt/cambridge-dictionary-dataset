import sqlite3

def fetch_cambridge_senses(cursor: sqlite3.Cursor, word: str) -> list[tuple[int, str, str]]:
    """Fetch Cambridge senses for a specific word display form."""
    cursor.execute("""
        SELECT s.id, e.pos, s.definition 
        FROM words w
        JOIN entries e ON w.id = e.word_id
        JOIN senses s ON e.id = s.entry_id
        WHERE w.display_form = ?
          AND e.dictionary_source = "Cambridge Advanced Learner's Dictionary"
          AND (s.phrase_title IS NULL OR s.phrase_title = '')
    """, (word,))
    return cursor.fetchall()

def find_polysemous_words(cursor: sqlite3.Cursor, limit: int = 100) -> list[str]:
    """Find words with >= 3 senses in CALD that also exist in WordNet (oewn:2024)."""
    import wn
    oewn = wn.Wordnet('oewn:2024')
    
    cursor.execute("""
        SELECT w.display_form, COUNT(s.id) as sense_count
        FROM words w
        JOIN entries e ON w.id = e.word_id
        JOIN senses s ON e.id = s.entry_id
        WHERE e.dictionary_source = "Cambridge Advanced Learner's Dictionary"
          AND (s.phrase_title IS NULL OR s.phrase_title = '')
        GROUP BY w.display_form
        HAVING sense_count >= 3
        ORDER BY sense_count DESC
    """)
    
    words = []
    for row in cursor.fetchall():
        word = row[0]
        if oewn.synsets(word):
            words.append(word)
            if len(words) >= limit:
                break
    return words

