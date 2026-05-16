from pydantic import BaseModel
from typing import List, Optional

class UserCreate(BaseModel):
    username: str
    password: str
    specialty: str

class UserLogin(BaseModel):
    username: str
    password: str

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: Optional[float] = None
    latency_ms: Optional[dict] = None

class Token(BaseModel):
    access_token: str
    token_type: str
    lawyer_id: str

class HealthStatus(BaseModel):
    status: str
    version: str
