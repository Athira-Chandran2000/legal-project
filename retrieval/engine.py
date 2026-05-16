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
    def __init__(self, embedding_model_name: str = "BAAI/bge-base-en-v1.5"):
        self.device = "cpu"
        print(f"Initializing RetrievalEngine with model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name, device=self.device)
        self.reranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="/tmp/flashrank")
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
        """Stable equal-weight RRF to prevent keyword bias noise."""
        fused_scores = {}
        # Dense ranks
        for rank, idx in enumerate(dense_results):
            fused_scores[idx] = fused_scores.get(idx, 0) + 1 / (k + rank + 1)
        # Sparse ranks (BM25) - Balanced at 1.0x
        sparse_indices = np.argsort(sparse_scores)[::-1][:20]
        for rank, idx in enumerate(sparse_indices):
            fused_scores[idx] = fused_scores.get(idx, 0) + 1 / (k + rank + 1)
        return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    def query(self, lawyer_id: str, query_text: str):
        """Deep Signal RAG with Query Expansion and Chain-of-Thought synthesis."""
        latency = {}
        start_total = time.time()

        # Step 0: Legal Query Expansion
        t0 = time.time()
        search_queries = [query_text]
        if self.groq_client:
            try:
                expand_resp = self.groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{
                        "role": "system", 
                        "content": "Rewrite this legal question using three alternative phrasings that use formal contract law terminology. Return only the 3 rephrased questions."
                    }, {"role": "user", "content": query_text}],
                    max_tokens=150,
                    temperature=0.1
                )
                variations = expand_resp.choices[0].message.content.split("\n")
                search_queries.extend([v.strip() for v in variations if v.strip()][:3])
            except: pass
        latency["query_expansion"] = (time.time() - t0) * 1000

        # Stage 7: Tenant Resource Loading
        try:
            t0 = time.time()
            index, bm25, chunks, metadata_db_path = self._load_tenant_resources(lawyer_id)
            latency["resource_loading"] = (time.time() - t0) * 1000
        except FileNotFoundError:
            return {"answer": "No documents found.", "sources": [], "latency": {"total": 0}}

        # Step 1: Broad Search with BGE Prefix
        t0 = time.time()
        # BGE Mandatory Prefix: "Represent this sentence for searching relevant passages: "
        prefixed_queries = [f"Represent this sentence for searching relevant passages: {q}" for q in search_queries]
        q_embeddings = self.embedding_model.encode(prefixed_queries, normalize_embeddings=True)
        # Average embeddings for query expansion consensus
        avg_embedding = np.mean(q_embeddings, axis=0, keepdims=True)
        
        dist, ind = index.search(avg_embedding, 20)
        dense_hits = ind[0].tolist()
        bm25_scores = bm25.get_scores(query_text.split())
        fused_results = self._rrf_fusion(dense_hits, bm25_scores)
        latency["hybrid_retrieval"] = (time.time() - t0) * 1000

        # Step 2: Reranking (High Signal Window)
        t0 = time.time()
        # Filter for valid indices only
        max_idx = len(chunks)
        top_20_indices = [idx for idx, score in fused_results[:20] if 0 <= idx < max_idx]
        
        try:
            passages = [{"id": chunks[idx]["chunk_id"], "text": chunks[idx]["text"]} for idx in top_20_indices]
            rerank_request = RerankRequest(query=query_text, passages=passages)
            reranked_results = self.reranker.rerank(rerank_request)
            top_5 = [res for res in reranked_results if res["score"] > 0.001][:5]
            # Map back to original indices for verification
            top_5_ids = [res["id"] for res in top_5]
            top_5_indices = [idx for idx in top_20_indices if chunks[idx]["chunk_id"] in top_5_ids]
        except Exception as e:
            print(f"Warning: Reranking failed (falling back to hybrid): {e}")
            top_5_indices = top_20_indices[:5]
            
        latency["reranking"] = (time.time() - t0) * 1000

        # Step 3: Verification & Parent-Context Injection
        t0 = time.time()
        verified_context = []
        sources = []
        conn = sqlite3.connect(metadata_db_path)
        cursor = conn.cursor()
        for idx in top_5_indices:
            cursor.execute("SELECT parent_text, doc_name, lawyer_id FROM chunks WHERE id = ?", (chunks[idx]["chunk_id"],))
            row = cursor.fetchone()
            if row and str(row[2]) == str(lawyer_id):
                if row[0] not in [c["text"] for c in verified_context]:
                    verified_context.append({"text": row[0], "doc": row[1]})
                    sources.append(row[1])
        conn.close()
        latency["verification"] = (time.time() - t0) * 1000

        # Stage 8: Chain-of-Thought Legal Synthesis
        t0 = time.time()
        context_str = "\n\n".join([f"Document [{c['doc']}]: {c['text']}" for c in verified_context])
        system_prompt = (
            "You are a professional legal assistant. Answer ONLY using the provided context. "
            "Follow this process:\n"
            "1. Identify the retrieved contract section that most directly addresses the question.\n"
            "2. Construct a comprehensive answer of at least three to five sentences.\n"
            "3. If the context partially addresses the question, provide what can be answered and explicitly identify what specific information is missing.\n"
            "4. Structure: Direct answer first, supporting evidence with [Document Name] citations second, and caveats third.\n"
            "5. If no information is found, say: 'Insufficient information in retrieved documents.'"
        )
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query_text}"}
                ],
                temperature=0.1
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Error during synthesis: {str(e)}"

        latency["generation"] = (time.time() - t0) * 1000
        latency["total"] = (time.time() - start_total) * 1000
        return {"answer": answer, "sources": list(set(sources)), "latency": latency}
