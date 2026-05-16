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

# Import tenant manager
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auth.tenant_manager import TenantManager

class RetrievalEngine:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.device = "cpu"
        print(f"Initializing RetrievalEngine with model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name, device=self.device)
        self.reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank")
        api_key = os.getenv("GROQ_API_KEY")
        self.groq_client = Groq(api_key=api_key) if api_key else None

    def _load_tenant_resources(self, lawyer_id: str):
        paths = TenantManager.get_tenant_index_paths(lawyer_id)
        if not os.path.exists(paths["faiss_index"]):
            raise FileNotFoundError(f"No documents for {lawyer_id}")
        index = faiss.read_index(paths["faiss_index"])
        with open(paths["bm25_pickle"], 'rb') as f:
            bm25 = pickle.load(f)
        with open(paths["chunks_json"], 'r') as f:
            chunks = json.load(f)
        return index, bm25, chunks, paths["metadata_db"]

    def _rrf_fusion(self, dense_results, sparse_scores, k=60):
        """Standard 50/50 RRF for stability."""
        fused_scores = {}
        for rank, idx in enumerate(dense_results):
            fused_scores[idx] = fused_scores.get(idx, 0) + 1 / (k + rank + 1)
        sparse_indices = np.argsort(sparse_scores)[::-1][:40]
        for rank, idx in enumerate(sparse_indices):
            fused_scores[idx] = fused_scores.get(idx, 0) + 1 / (k + rank + 1)
        return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    def query(self, lawyer_id: str, query_text: str):
        """Stable, direct RAG pipeline with Parent-Context injection."""
        latency = {}
        start_total = time.time()

        # Stage 7: Tenant Resource Loading
        try:
            t0 = time.time()
            index, bm25, chunks, metadata_db_path = self._load_tenant_resources(lawyer_id)
            latency["resource_loading"] = (time.time() - t0) * 1000
        except FileNotFoundError:
            return {"answer": "No documents found.", "sources": [], "latency": {"total": 0}}

        # Step 1: Hybrid Retrieval (Pure Query)
        t0 = time.time()
        q_emb = self.embedding_model.encode([query_text], normalize_embeddings=True)
        dist, ind = index.search(q_emb, 30)
        dense_hits = ind[0].tolist()
        bm25_scores = bm25.get_scores(query_text.split())
        fused_results = self._rrf_fusion(dense_hits, bm25_scores)
        latency["hybrid_retrieval"] = (time.time() - t0) * 1000

        # Step 2: Reranking
        t0 = time.time()
        top_20_indices = [idx for idx, score in fused_results[:20]]
        passages = [{"id": chunks[idx]["chunk_id"], "text": chunks[idx]["text"]} for idx in top_20_indices if idx < len(chunks)]
        rerank_request = RerankRequest(query=query_text, passages=passages)
        reranked_results = self.reranker.rerank(rerank_request)
        top_5 = [res for res in reranked_results if res["score"] > 0.001][:5]
        latency["reranking"] = (time.time() - t0) * 1000

        # Step 3: Isolation & Parent-Context Swap
        t0 = time.time()
        verified_context = []
        sources = []
        conn = sqlite3.connect(metadata_db_path)
        cursor = conn.cursor()
        for hit in top_5:
            cursor.execute("SELECT parent_text, doc_name, lawyer_id FROM chunks WHERE id = ?", (hit["id"],))
            row = cursor.fetchone()
            if row and str(row[2]) == str(lawyer_id):
                if row[0] not in [c["text"] for c in verified_context]:
                    verified_context.append({"text": row[0], "doc": row[1]})
                    sources.append(row[1])
        conn.close()
        latency["verification"] = (time.time() - t0) * 1000

        # Stage 8: Conversational Generation
        t0 = time.time()
        context_str = "\n\n".join([f"Document [{c['doc']}]: {c['text']}" for c in verified_context])
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{
                    "role": "system", 
                    "content": "You are a direct legal assistant. Answer conversational and clearly based strictly on the provided documents. Always cite [Document Name]. If info is missing, say you don't have access to it."
                }, {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query_text}"}],
                temperature=0.1
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Error: {str(e)}"

        latency["generation"] = (time.time() - t0) * 1000
        latency["total"] = (time.time() - start_total) * 1000
        return {"answer": answer, "sources": list(set(sources)), "latency": latency}
