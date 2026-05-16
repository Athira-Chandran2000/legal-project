import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from retrieval.engine import RetrievalEngine
from auth.database import SessionLocal
from auth.models import Lawyer

def test_query():
    load_dotenv()
    db = SessionLocal()
    lawyer = db.query(Lawyer).filter(Lawyer.username == "lawyer1_nda").first()
    if not lawyer:
        print("Lawyer not found. Run setup scripts first.")
        return

    engine = RetrievalEngine()
    question = "What is the governing law mentioned in the Alpha Corp agreement"
    print(f"\nQuerying: {question}")
    
    result = engine.query(lawyer.id, question)
    
    print("\n--- ANSWER ---")
    print(result["answer"])
    print("\n--- SOURCES ---")
    print(result["sources"])
    print("\n--- LATENCY ---")
    print(result["latency"])

    db.close()

if __name__ == "__main__":
    test_query()
