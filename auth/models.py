from sqlalchemy import Column, String, DateTime
from datetime import datetime
import uuid
from .database import Base

class Lawyer(Base):
    __tablename__ = "lawyers"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    specialty = Column(String) # e.g., "NDA", "Employment", etc.
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Lawyer(username='{self.username}', specialty='{self.specialty}')>"
