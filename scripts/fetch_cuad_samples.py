import torch
import os
import sys
import json
from datasets import load_dataset

# Fix for WinError 1114
if os.name == 'nt':
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.exists(torch_lib_path):
        os.add_dll_directory(torch_lib_path)

# Import internal modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingestion.pipeline import IngestionPipeline
from auth.database import SessionLocal
from auth.models import Lawyer
from auth.tenant_manager import TenantManager

def fetch_real_cuad():
    try:
        # We'll load the filtered Parquet version of CUAD
        dataset = load_dataset("alex-apostolo/filtered-cuad", split="train")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return

    db = SessionLocal()
    lawyers = db.query(Lawyer).all()
    
    if not lawyers:
        print("No lawyers found. Run setup_demo.py first.")
        return

    pipeline = IngestionPipeline()
    
    # We will pick 30 unique contracts from the dataset for a robust demo
    unique_contracts = {}
    for item in dataset:
        doc_name = item.get("title", "Unknown_Contract")
        if doc_name not in unique_contracts and item.get("context"):
            unique_contracts[doc_name] = item["context"]
        if len(unique_contracts) >= 30:
            break

    contract_list = list(unique_contracts.items())
    
    # Distribute real contracts to our demo lawyers (10 each)
    for i, lawyer in enumerate(lawyers):
        start_idx = (i * 10) % len(contract_list)
        lawyer_docs = []
        
        print(f"\nProcessing 10 docs for {lawyer.username}...")
        for offset in range(10):
            idx = (start_idx + offset) % len(contract_list)
            name, text = contract_list[idx]
            
            # Save the raw file to the docs folder so it's visible to the user
            tenant_root = TenantManager.get_tenant_path(lawyer.id)
            docs_dir = os.path.join(tenant_root, "docs")
            os.makedirs(docs_dir, exist_ok=True)
            
            file_path = os.path.join(docs_dir, f"{name}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            lawyer_docs.append({"doc_name": name, "text": text})
            print(f"Saved real CUAD contract to {lawyer.username}: {name}")

        print(f"Indexing real documents for {lawyer.username}...")
        pipeline.process_lawyer_documents(lawyer.id, lawyer_docs)

    db.close()
    print("\nSuccess! Real CUAD contracts are now in the docs folders and indexed.")

if __name__ == "__main__":
    fetch_real_cuad()
