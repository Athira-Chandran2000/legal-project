import os
import sys
from typing import List, Dict, Any
from datasets import load_dataset

# Import internal modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingestion.pipeline import IngestionPipeline
from auth.database import SessionLocal
from auth.models import Lawyer

def ingest_samples():
    print("Ingesting sample legal documents for demo lawyers...")
    db = SessionLocal()
    lawyers = db.query(Lawyer).all()
    
    if not lawyers:
        print("No lawyers found. Run setup_demo.py first.")
        db.close()
        return

    pipeline = IngestionPipeline()

    # Load CUAD dataset from the Hub for realistic scale
    try:
        print("Loading CUAD dataset from Hugging Face Hub...")
        # CUAD v1 is available as atticus_legal/cuad
        dataset = load_dataset("atticus_legal/cuad", split="train")
        # In the Hub version, 'context' and 'qas' are usually nested or in columns
        # We'll take unique contexts to get our 500 documents
        cuad_docs = []
        seen_contexts = set()
        for row in dataset:
            ctx = row.get("context", "")
            if ctx and ctx not in seen_contexts:
                cuad_docs.append({"doc_name": f"CUAD_{len(cuad_docs)}", "text": ctx})
                seen_contexts.add(ctx)
                if len(cuad_docs) >= 500: break
        print(f"Streaming {len(cuad_docs)} unique contracts from CUAD.")
    except Exception as e:
        print(f"Warning: Hub loading failed. Error: {e}")
        cuad_docs = []

    sample_docs = {
        "NDA": [{"doc_name": "NDA_Alpha.txt", "text": "Confidentiality agreement..."}, {"doc_name": "NDA_Beta.txt", "text": "Mutual NDA..."}],
        "IP Licensing": [{"doc_name": "SLA.txt", "text": "Software License..."}],
        "Real Estate": [{"doc_name": "Lease.txt", "text": "Commercial Lease..."}]
    }

    for lawyer in lawyers:
        specialty = lawyer.specialty
        
        # Pull 50 docs for this lawyer based on specialty or general sampling
        docs_to_index = []
        if cuad_docs:
            # For demo, we just take segments of the CUAD corpus
            start_idx = (lawyers.index(lawyer) * 50) % len(cuad_docs)
            for i in range(start_idx, start_idx + 50):
                doc = cuad_docs[i % len(cuad_docs)]
                docs_to_index.append({
                    "doc_name": doc.get("qas", [{}])[0].get("id", f"CUAD_{i}"),
                    "text": doc.get("context", "")
                })
        else:
            # Fallback to hardcoded samples
            docs_to_index = sample_docs.get(specialty, [])

        if docs_to_index:
            print(f"Indexing {len(docs_to_index)} documents for {lawyer.username} ({specialty})...")
            pipeline.process_lawyer_documents(lawyer.id, docs_to_index)
        else:
            print(f"No sample documents for specialty {specialty} (Lawyer: {lawyer.username})")

    db.close()
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_samples()
