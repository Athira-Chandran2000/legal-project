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

    def _weighted_rrf(self, dense_results, sparse_scores, k=60):
        """Modified RRF to favor BM25 for legal jargon."""
        fused_scores = {}
        # Dense ranks (Semantic)
        for rank, idx in enumerate(dense_results):
            fused_scores[idx] = fused_scores.get(idx, 0) + 1 / (k + rank + 1)
        # Sparse ranks (Keywords) - Weighted 2.0x for legal precision
        sparse_indices = np.argsort(sparse_scores)[::-1][:60]
        for rank, idx in enumerate(sparse_indices):
            fused_scores[idx] = fused_scores.get(idx, 0) + (2.0 / (k + rank + 1))
        return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    def query(self, lawyer_id: str, query_text: str):
        """Executes the full isolated RAG pipeline with Multi-Query Expansion and Deep Recall."""
        latency = {}
        start_total = time.time()

        # Step 0: Multi-Query Expansion to fix Recall@5 blind spots
        t0 = time.time()
        queries = [query_text]
        if self.groq_client:
            try:
                expand_resp = self.groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{
                        "role": "system", 
                        "content": "Generate 2 different legal search queries and one hypothetical answer for this question. Output 3 lines."
                    }, {"role": "user", "content": query_text}],
                    max_tokens=150,
                    temperature=0.2
                )
                variations = expand_resp.choices[0].message.content.split("\n")
                queries.extend([v.strip() for v in variations if v.strip()][:3])
            except: pass
        latency["query_expansion"] = (time.time() - t0) * 1000

        # Stage 7: Tenant Resource Loading
        try:
            t0 = time.time()
            index, bm25, chunks, metadata_db_path = self._load_tenant_resources(lawyer_id)
            latency["resource_loading"] = (time.time() - t0) * 1000
        except FileNotFoundError:
            return {"answer": "I don't have access to documents for your account.", "sources": [], "latency": {"total": 0}}

        # Step 1 & 2: Extreme Hybrid Retrieval (k=100)
        t0 = time.time()
        all_dense_hits = []
        for q in queries:
            q_emb = self.embedding_model.encode([q], normalize_embeddings=True)
            dist, ind = index.search(q_emb, 40)
            all_dense_hits.extend(ind[0].tolist())
        
        unique_dense = list(set(all_dense_hits))
        bm25_scores = bm25.get_scores(query_text.split())
        fused_results = self._weighted_rrf(unique_dense, bm25_scores)
        latency["hybrid_retrieval"] = (time.time() - t0) * 1000

        # Step 3: Flashrank Reranking (Broad search, narrow feed)
        t0 = time.time()
        top_40_indices = [idx for idx, score in fused_results[:40]]
        top_40_chunks = [chunks[idx] for idx in top_40_indices if idx < len(chunks)]
        passages = [{"id": c["chunk_id"], "text": c["text"], "meta": c} for c in top_40_chunks]
        rerank_request = RerankRequest(query=query_text, passages=passages)
        reranked_results = self.reranker.rerank(rerank_request)
        top_5 = [res for res in reranked_results if res["score"] > 0.005][:5]
        latency["reranking"] = (time.time() - t0) * 1000

        # Step 4: Verification & Isolation Firewall
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

        # Stage 8: Generation (Conversational ChatGPT Style)
        t0 = time.time()
        context_str = "\n\n".join([f"Document [{c['doc']}]: {c['text']}" for c in verified_context])
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{
                    "role": "system", 
                    "content": "You are a professional legal assistant. Provide helpful, conversational answers based strictly on the provided documents. Cite sources as [Document Name]. If information is missing, state so clearly."
                }, {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query_text}"}],
                temperature=0.1
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Error during generation: {str(e)}"

        latency["generation"] = (time.time() - t0) * 1000
        latency["total"] = (time.time() - start_total) * 1000
        return {"answer": answer, "sources": list(set(sources)), "latency": latency}
