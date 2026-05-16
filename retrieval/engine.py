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
        
        # Dense ranks (Semantic)
        for rank, idx in enumerate(dense_results):
            fused_scores[idx] = fused_scores.get(idx, 0) + 1 / (k + rank + 1)
            
        # Sparse ranks (Keywords) - Weighted higher (x1.5) to favor legal jargon
        sparse_indices = np.argsort(sparse_scores)[::-1][:40]
        for rank, idx in enumerate(sparse_indices):
            fused_scores[idx] = fused_scores.get(idx, 0) + (1.5 / (k + rank + 1))
            
        return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    def query(self, lawyer_id: str, query_text: str):
        """Executes the full isolated RAG pipeline with Zero-Leak guarantee."""
        latency = {}
        start_total = time.time()

        # Step 0: HyDE (Hypothetical Document Embedding)
        t0 = time.time()
        hyde_answer = ""
        if self.groq_client:
            try:
                hyde_resp = self.groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": f"Briefly answer this legal question: {query_text}"}],
                    max_tokens=100,
                    temperature=0.1
                )
                hyde_answer = hyde_resp.choices[0].message.content
            except:
                hyde_answer = query_text
        else:
            hyde_answer = query_text
        latency["hyde_generation"] = (time.time() - t0) * 1000

        # Stage 7: Tenant Resource Loading (STRICT PHYSICAL ISOLATION)
        # We load a FRESH handle for every query to prevent cross-tenant bleeding
        try:
            t0 = time.time()
            paths = TenantManager.get_tenant_index_paths(lawyer_id)
            if not os.path.exists(paths["faiss_index"]):
                raise FileNotFoundError
                
            index = faiss.read_index(paths["faiss_index"])
            with open(paths["bm25_pickle"], 'rb') as f:
                bm25 = pickle.load(f)
            with open(paths["chunks_json"], 'r') as f:
                chunks = json.load(f)
            latency["resource_loading"] = (time.time() - t0) * 1000
        except FileNotFoundError:
            return {"answer": "I don't have access to any documents for your account yet.", "sources": [], "latency": {"total": 0}}

        # Step 1 & 2: Hybrid Retrieval
        t0 = time.time()
        query_embedding = self.embedding_model.encode([hyde_answer], normalize_embeddings=True)
        distances, indices = index.search(query_embedding, 40) 
        dense_hits = indices[0].tolist()
        
        tokenized_query = query_text.split()
        bm25_scores = bm25.get_scores(tokenized_query)
        fused_results = self.rrf_fusion(dense_hits, bm25_scores)
        latency["hybrid_retrieval"] = (time.time() - t0) * 1000

        # Step 3: Flashrank Reranking
        t0 = time.time()
        top_25_indices = [idx for idx, score in fused_results[:25]]
        top_25_chunks = [chunks[idx] for idx in top_25_indices if idx < len(chunks)]
        passages = [{"id": c["chunk_id"], "text": c["text"], "meta": c} for c in top_25_chunks]
        rerank_request = RerankRequest(query=query_text, passages=passages)
        reranked_results = self.reranker.rerank(rerank_request)
        
        # Take Top 6 (Clean context window)
        top_6 = [res for res in reranked_results if res["score"] > 0.01][:6]
        latency["reranking"] = (time.time() - t0) * 1000

        # Step 4: Verification & Swap (THE FIREWALL)
        t0 = time.time()
        verified_context = []
        sources = []
        conn = sqlite3.connect(paths["metadata_db"])
        cursor = conn.cursor()
        for hit in top_6:
            # We strictly verify that every retrieved chunk belongs to THIS lawyer_id
            cursor.execute("SELECT parent_text, doc_name, lawyer_id FROM chunks WHERE id = ?", (hit["id"],))
            row = cursor.fetchone()
            if row and str(row[2]) == str(lawyer_id):
                if row[0] not in [c["text"] for c in verified_context]:
                    verified_context.append({"text": row[0], "doc": row[1]})
                    sources.append(row[1])
        conn.close()
        latency["verification"] = (time.time() - t0) * 1000

        # Stage 8: Generation (CONVERSATIONAL STYLE)
        t0 = time.time()
        context_str = "\n\n".join([f"Document [{c['doc']}]: {c['text']}" for c in verified_context])
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            "You are a professional legal assistant. Your goal is to provide helpful, conversational, and direct answers "
                            "based strictly on the provided documents. Speak naturally like ChatGPT, but always include citations like [Document Name] "
                            "immediately after stating a fact from that source. "
                            "If the user asks about a document that is not in your context (like a software license or real estate when you only have NDAs), "
                            "you MUST state that you do not have access to that information."
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
