import os
import sys
from sqlalchemy.orm import Session

# Import internal modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auth.database import SessionLocal, engine, Base
from auth.models import Lawyer
from auth.security import get_password_hash
from auth.tenant_manager import TenantManager

def setup_demo():
    print("Initializing Demo Data...")
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    demo_lawyers = [
        {"username": "lawyer1_nda", "password": "password123", "specialty": "NDA"},
        {"username": "lawyer3_ip", "password": "password123", "specialty": "IP Licensing"},
        {"username": "lawyer5_realestate", "password": "password123", "specialty": "Real Estate"}
    ]

    for lawyer_data in demo_lawyers:
        existing = db.query(Lawyer).filter(Lawyer.username == lawyer_data["username"]).first()
        if not existing:
            new_lawyer = Lawyer(
                username=lawyer_data["username"],
                hashed_password=get_password_hash(lawyer_data["password"]),
                specialty=lawyer_data["specialty"]
            )
            db.add(new_lawyer)
            db.commit()
            db.refresh(new_lawyer)
            TenantManager.initialize_tenant(new_lawyer.id)
            print(f"Created demo lawyer: {lawyer_data['username']} (ID: {new_lawyer.id})")
        else:
            print(f"Demo lawyer {lawyer_data['username']} already exists.")

    db.close()
    print("Demo setup complete.")

if __name__ == "__main__":
    setup_demo()
