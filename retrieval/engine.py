import torch
import os
import time
import pickle
import sqlite3
import json
import faiss
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from flashrank import Ranker, RerankRequest
from groq import Groq
import torch

# Import tenant manager
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auth.tenant_manager import TenantManager

class RetrievalEngine:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.device = "cpu" # Force CPU for stability on Windows
        print(f"Initializing RetrievalEngine with model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name, device=self.device)
        self.reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank")
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            self.groq_client = Groq(api_key=api_key)
        else:
            self.groq_client = None
            print("WARNING: GROQ_API_KEY not found. Generation phase will be disabled.")

    def _load_tenant_resources(self, lawyer_id: str):
        """Loads all index files for a specific lawyer."""
        paths = TenantManager.get_tenant_index_paths(lawyer_id)
        
        if not os.path.exists(paths["faiss_index"]):
            raise FileNotFoundError(f"No documents found for lawyer {lawyer_id}")

        index = faiss.read_index(paths["faiss_index"])
        
        with open(paths["bm25_pickle"], 'rb') as f:
            bm25 = pickle.load(f)
            
        with open(paths["chunks_json"], 'r') as f:
            chunks = json.load(f)
            
        return index, bm25, chunks, paths["metadata_db"]

    def rrf_fusion(self, dense_results: List[int], sparse_scores: np.ndarray, k: int = 60):
        """Reciprocal Rank Fusion."""
        # This is a simplified version for the demo
        # In practice, we combine ranks from both lists
        fused_scores = {}
        
        # Dense ranks
        for rank, idx in enumerate(dense_results):
            fused_scores[idx] = fused_scores.get(idx, 0) + 1 / (k + rank + 1)
            
        # Sparse ranks (top indices from bm25 scores)
        sparse_indices = np.argsort(sparse_scores)[::-1][:30]
        for rank, idx in enumerate(sparse_indices):
            fused_scores[idx] = fused_scores.get(idx, 0) + 1 / (k + rank + 1)
            
        return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    def query(self, lawyer_id: str, query_text: str):
        """Executes the full multi-tenant RAG pipeline with latency tracking."""
        latency = {}
        start_total = time.time()

        # Stage 7: Tenant Resolution
        try:
            t0 = time.time()
            index, bm25, chunks, metadata_db_path = self._load_tenant_resources(lawyer_id)
            latency["resource_loading"] = (time.time() - t0) * 1000
        except FileNotFoundError:
            return {"answer": "No documents found.", "sources": [], "latency": {"total": 0}}

        # Step 1 & 2: Hybrid Retrieval
        t0 = time.time()
        query_embedding = self.embedding_model.encode([query_text], normalize_embeddings=True)
        distances, indices = index.search(query_embedding, 30)
        dense_hits = indices[0].tolist()
        
        tokenized_query = query_text.split()
        bm25_scores = bm25.get_scores(tokenized_query)
        fused_results = self.rrf_fusion(dense_hits, bm25_scores)
        latency["hybrid_retrieval"] = (time.time() - t0) * 1000

        # Step 3: Flashrank Reranking
        t0 = time.time()
        top_20_indices = [idx for idx, score in fused_results[:20]]
        top_20_chunks = [chunks[idx] for idx in top_20_indices if idx < len(chunks)]
        passages = [{"id": c["chunk_id"], "text": c["text"], "meta": c} for c in top_20_chunks]
        rerank_request = RerankRequest(query=query_text, passages=passages)
        reranked_results = self.reranker.rerank(rerank_request)
        
        # FIX: Filter by threshold and take only top 3 to reduce noise injection
        # Flashrank scores are usually 0-1. 0.1 is a safe but strict floor for this model.
        top_3 = [res for res in reranked_results if res["score"] > 0.1][:3]
        latency["reranking"] = (time.time() - t0) * 1000

        # Step 4: Verification & Swap
        t0 = time.time()
        verified_context = []
        sources = []
        conn = sqlite3.connect(metadata_db_path)
        cursor = conn.cursor()
        for hit in top_3:
            cursor.execute("SELECT parent_text, doc_name, lawyer_id FROM chunks WHERE id = ?", (hit["id"],))
            row = cursor.fetchone()
            if row and row[2] == lawyer_id:
                if row[0] not in [c["text"] for c in verified_context]:
                    verified_context.append({"text": row[0], "doc": row[1]})
                    sources.append(row[1])
        conn.close()
        latency["verification"] = (time.time() - t0) * 1000

        # Stage 8: Generation
        t0 = time.time()
        context_str = "\n\n".join([f"Source [{c['doc']}]: {c['text']}" for c in verified_context])
        
        if not self.groq_client:
            latency["generation"] = 0
            latency["total"] = (time.time() - start_total) * 1000
            return {"answer": "Generation disabled: GROQ_API_KEY missing.", "sources": list(set(sources)), "latency": latency}

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            "You are an expert legal counsel. Your goal is to provide a direct, professional, and synthesized answer to the user's specific question using the provided context. "
                            "Do not use defensive filler like 'According to the context'. Instead, state the facts directly as they appear in the documents. "
                            "If a direct answer isn't explicitly written but can be logically inferred from the facts provided (e.g., calculating dates or combining clauses), you MUST provide that logical inference. "
                            "Keep answers concise and high-impact. Always cite the document source names as your authority."
                        )
                    },
                    {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query_text}"}
                ],
                temperature=0.1
            )
            answer = response.choices[0].message.content
        except Exception as e:
            print(f"Groq API Error: {e}")
            answer = f"I'm sorry, I encountered an error during generation: {str(e)}"

        latency["generation"] = (time.time() - t0) * 1000
        latency["total"] = (time.time() - start_total) * 1000

        return {
            "answer": answer,
            "sources": list(set(sources)),
            "latency": latency
        }
