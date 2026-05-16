---
title: LexGuard Legal RAG
emoji: ⚖️
colorFrom: blue
colorTo: gold
sdk: docker
pinned: false
app_port: 7860
---

# Legal Multi-Tenant RAG System

A production-grade, isolated RAG system for legal professionals. This system ensures 100% tenant isolation by using separate indexes per lawyer.

## Architecture
- **Isolation**: Separate FAISS, SQLite, and BM25 indexes per lawyer.
- **Retrieval**: Hybrid Search (FAISS + BM25) + RRF + Flashrank Reranking.
- **LLM**: Groq (Llama 3.1 8B) for low-latency, grounded legal answers.
- **Auth**: JWT-based authentication where index selection is driven by verified token claims.

## Project Structure
- `/app`: FastAPI backend and schemas.
- `/auth`: JWT security, central database, and tenant isolation logic.
- `/ingestion`: Pipeline for processing CUAD and custom uploads.
- `/retrieval`: Hybrid retrieval engine with secondary verification.
- `/scripts`: Evaluation suite, demo setup, and maintenance.
- `/data/tenants`: Root directory for isolated lawyer data.

## Demo Credentials
| Lawyer Role | Username | Password |
|---|---|---|
| NDA Specialist | `lawyer1_nda` | `password123` |
| IP Specialist | `lawyer3_ip` | `password123` |
| Real Estate | `lawyer5_realestate` | `password123` |

## Getting Started
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure Environment**:
   - Create a `.env` file (template provided).
   - Add your `GROQ_API_KEY`.
3. **Setup Demo Data**:
   ```bash
   python scripts/setup_demo.py
   ```
3. **Run API**:
   ```bash
   uvicorn app.main:app --reload
   ```
4. **Access Swagger UI**: `http://localhost:8000/docs`

## Evaluation
- **Isolation Pass Rate**: 100% (Mandatory for deployment).
- **Quality**: RAGAS metrics (Faithfulness, Relevancy, Precision).
- **Tracking**: Remote MLflow via DagsHub.

## Deployment
- **Platform**: Hugging Face Spaces (Docker).
- **CI/CD**: GitHub Actions with automated isolation testing.
