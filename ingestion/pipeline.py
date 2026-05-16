import torch
import os
import json
import pickle
import sqlite3
import numpy as np
import faiss
from typing import List, Dict, Any
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
# Fix for WinError 1114 on some Windows systems
if os.name == 'nt':
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.exists(torch_lib_path):
        os.add_dll_directory(torch_lib_path)

# Import tenant manager
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auth.tenant_manager import TenantManager

class IngestionPipeline:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        print(f"Initializing IngestionPipeline with model: {embedding_model_name}")
        self.device = "cpu" # Force CPU for stability on Windows
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name, device=self.device)
            print("Embedding model loaded successfully.")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            raise e
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=60,
            separators=["\n\n", "\n", " ", ""]
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=150,
            chunk_overlap=30,
            separators=["\n\n", "\n", " ", ""]
        )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extracts text from a PDF file using PyMuPDF."""
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    def process_lawyer_documents(self, lawyer_id: str, documents: List[Dict[str, Any]] = None):
        """
        Processes and indexes documents for a specific lawyer.
        If documents is None, it reads all files from the lawyer's docs folder.
        """
        tenant_paths = TenantManager.get_tenant_index_paths(lawyer_id)
        tenant_root = TenantManager.get_tenant_path(lawyer_id)
        docs_dir = os.path.join(tenant_root, "docs")
        
        if documents is None:
            documents = []
            if os.path.exists(docs_dir):
                for filename in os.listdir(docs_dir):
                    if filename.endswith(".pdf"):
                        text = self.extract_text_from_pdf(os.path.join(docs_dir, filename))
                        documents.append({"doc_name": filename, "text": text})
                    elif filename.endswith(".txt"):
                        with open(os.path.join(docs_dir, filename), 'r', encoding='utf-8') as f:
                            documents.append({"doc_name": filename, "text": f.read()})

        if not documents:
            print(f"No documents to index for {lawyer_id}")
            return 0

        TenantManager.initialize_tenant(lawyer_id)

        all_chunks = []
        child_embeddings = []
        
        # Step 1: Chunking
        for doc in documents:
            doc_name = doc.get("doc_name", "Unknown")
            full_text = doc.get("text", "")
            
            # Parent chunks
            parents = self.parent_splitter.create_documents([full_text])
            for p_idx, parent in enumerate(parents):
                parent_id = f"{lawyer_id}_{doc_name}_p{p_idx}"
                
                # Child chunks
                children = self.child_splitter.create_documents([parent.page_content])
                for c_idx, child in enumerate(children):
                    chunk_id = f"{parent_id}_c{c_idx}"
                    chunk_data = {
                        "chunk_id": chunk_id,
                        "parent_id": parent_id,
                        "parent_text": parent.page_content,
                        "text": child.page_content,
                        "doc_name": doc_name,
                        "lawyer_id": lawyer_id
                    }
                    all_chunks.append(chunk_data)
        
        # Save chunks JSON
        with open(tenant_paths["chunks_json"], 'w') as f:
            json.dump(all_chunks, f)

        # Step 2: Embedding (Batch processing on GPU)
        texts_to_embed = [c["text"] for c in all_chunks]
        if texts_to_embed:
            embeddings = self.embedding_model.encode(
                texts_to_embed, 
                batch_size=256, 
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Step 3: FAISS Index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings)
            faiss.write_index(index, tenant_paths["faiss_index"])

        # Step 4: SQLite Metadata
        conn = sqlite3.connect(tenant_paths["metadata_db"])
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS chunks 
                          (id TEXT PRIMARY KEY, parent_id TEXT, parent_text TEXT, 
                           text TEXT, doc_name TEXT, lawyer_id TEXT)''')
        
        # Clear existing data for clean re-indexing
        cursor.execute("DELETE FROM chunks")
        conn.commit()
        
        for chunk in all_chunks:
            cursor.execute("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
                           (chunk["chunk_id"], chunk["parent_id"], chunk["parent_text"],
                            chunk["text"], chunk["doc_name"], chunk["lawyer_id"]))
        conn.commit()
        conn.close()

        # Step 5: BM25 Index
        if all_chunks:
            tokenized_corpus = [c["text"].split() for c in all_chunks]
            bm25 = BM25Okapi(tokenized_corpus)
            with open(tenant_paths["bm25_pickle"], 'wb') as f:
                pickle.dump(bm25, f)

        print(f"Completed indexing for {lawyer_id}. Total chunks: {len(all_chunks)}")
        return len(all_chunks)
