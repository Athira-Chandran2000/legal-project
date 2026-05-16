import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from typing import List

# Internal imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth.database import get_db, engine, Base
from auth.models import Lawyer
from auth.security import verify_password, get_password_hash, create_access_token, decode_access_token
from auth.tenant_manager import TenantManager
from retrieval.engine import RetrievalEngine
from ingestion.pipeline import IngestionPipeline
from .schemas import UserCreate, UserLogin, QueryRequest, QueryResponse, Token, HealthStatus

from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

# Create database tables
Base.metadata.create_all(bind=engine)

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="LexGuard | Secure Legal RAG",
    description="Isolated RAG system for lawyers with hybrid retrieval and reranking.",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    return FileResponse("app/static/index.html")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Initialize engines
retrieval_engine = None
ingestion_pipeline = None

@app.on_event("startup")
async def startup_event():
    global retrieval_engine, ingestion_pipeline
    
    # Ensure demo data is initialized for the deployment environment
    try:
        print("Running one-time demo setup...")
        from scripts.setup_demo import setup_demo
        from scripts.ingest_samples import ingest_samples
        setup_demo()
        ingest_samples()
    except Exception as e:
        print(f"Startup setup skipped or failed: {e}")
        
    retrieval_engine = RetrievalEngine()
    ingestion_pipeline = IngestionPipeline()

# Dependency to get current lawyer
async def get_current_lawyer(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    lawyer_id = decode_access_token(token)
    if not lawyer_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    lawyer = db.query(Lawyer).filter(Lawyer.id == lawyer_id).first()
    if not lawyer:
        raise HTTPException(status_code=404, detail="Lawyer not found")
    return lawyer

@app.post("/register", status_code=status.HTTP_201_CREATED)
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(Lawyer).filter(Lawyer.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(user.password)
    new_lawyer = Lawyer(
        username=user.username,
        hashed_password=hashed_password,
        specialty=user.specialty
    )
    db.add(new_lawyer)
    db.commit()
    db.refresh(new_lawyer)
    
    # Initialize tenant directory structure
    TenantManager.initialize_tenant(new_lawyer.id)
    
    return {"message": "Lawyer registered successfully", "lawyer_id": new_lawyer.id}

@app.post("/login", response_model=Token)
def login(user: UserLogin, db: Session = Depends(get_db)):
    lawyer = db.query(Lawyer).filter(Lawyer.username == user.username).first()
    if not lawyer or not verify_password(user.password, lawyer.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    
    access_token = create_access_token(data={"lawyer_id": lawyer.id})
    return {"access_token": access_token, "token_type": "bearer", "lawyer_id": lawyer.id}

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...), 
    current_lawyer: Lawyer = Depends(get_current_lawyer)
):
    # Stage 9: Document Upload Endpoint
    tenant_path = TenantManager.get_tenant_path(current_lawyer.id)
    docs_dir = os.path.join(tenant_path, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    
    file_path = os.path.join(docs_dir, file.filename)
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Trigger incremental indexing for this lawyer
    num_chunks = ingestion_pipeline.process_lawyer_documents(current_lawyer.id)
    
    return {
        "message": f"File {file.filename} uploaded and indexed successfully.",
        "chunks": num_chunks
    }

@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest, 
    current_lawyer: Lawyer = Depends(get_current_lawyer)
):
    try:
        result = retrieval_engine.query(current_lawyer.id, request.question)
        
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "latency": result["latency"]
        }
    except Exception as e:
        print(f"Query error: {e}")
        return {
            "answer": f"Error: {str(e)}",
            "sources": [],
            "latency_ms": {"total": 0}
        }

@app.get("/smoke-test")
async def run_smoke_test(current_lawyer: Lawyer = Depends(get_current_lawyer)):
    # Only allow certain users or just for demo purposes
    from tests.test_isolation import test_tenant_isolation_logic
    try:
        test_tenant_isolation_logic()
        return {"status": "passed", "message": "Isolation verified: No cross-tenant leakage detected."}
    except Exception as e:
        return {"status": "failed", "message": str(e)}

@app.get("/health", response_model=HealthStatus)
def health():
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
