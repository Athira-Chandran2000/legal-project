import sqlite3
import json
import os

def diagnose():
    db_path = "data/tenants/47c20d52-e7ea-4683-aa96-2a78c0d623f0/metadata.db"
    eval_path = "data/eval_set.json"
    
    if not os.path.exists(db_path):
        print(f"DB not found at {db_path}")
        return
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT doc_name FROM chunks LIMIT 5")
    db_docs = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    print(f"Sample docs in DB: {db_docs}")
    
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            eval_set = json.load(f)
        eval_docs = [item["doc_name"] for item in eval_set[:5]]
        print(f"Sample docs in Eval Set: {eval_docs}")
        
        # Check for intersection
        matches = set(db_docs).intersection(set(eval_docs))
        print(f"Direct matches found: {len(matches)}")

if __name__ == "__main__":
    diagnose()
