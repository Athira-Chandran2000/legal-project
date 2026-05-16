import os
import sys
import httpx
import pytest
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from auth.database import SessionLocal, Base, engine
from auth.models import Lawyer

client = TestClient(app)

def test_tenant_isolation_logic():
    """
    Verifies that Lawyer A cannot access Lawyer B's data 
    even if they have a valid token.
    """
    with TestClient(app) as client:
        # 1. Login as Lawyer 1 (NDA)
        response_1 = client.post("/login", json={"username": "lawyer1_nda", "password": "password123"})
        assert response_1.status_code == 200
        token_1 = response_1.json()["access_token"]
        lawyer_1_id = response_1.json()["lawyer_id"]

        # 2. Login as Lawyer 3 (IP)
        response_3 = client.post("/login", json={"username": "lawyer3_ip", "password": "password123"})
        assert response_3.status_code == 200
        token_3 = response_3.json()["access_token"]
        lawyer_3_id = response_3.json()["lawyer_id"]

        # 3. Verify IDs are different
        assert lawyer_1_id != lawyer_3_id

        # 4. Query as Lawyer 1 (NDA) for NDA info
        query_resp_1 = client.post(
            "/query", 
            json={"question": "What is the term of the NDA?"},
            headers={"Authorization": f"Bearer {token_1}"}
        )
        assert query_resp_1.status_code == 200
        # Should NOT say "No documents found" now
        assert "No documents found" not in query_resp_1.json()["answer"]
        assert len(query_resp_1.json()["sources"]) > 0
        print(f"Lawyer 1 (NDA) query success. Sources: {query_resp_1.json()['sources']}")

        # 5. Query as Lawyer 1 for IP info (should NOT find IP docs)
        query_resp_leak = client.post(
            "/query", 
            json={"question": "What are the software royalties?"},
            headers={"Authorization": f"Bearer {token_1}"}
        )
        assert query_resp_leak.status_code == 200
        # Since Lawyer 1 doesn't have the IP doc, it should find nothing related to it
        # The answer might be "Context retrieved successfully but generation disabled" if no key,
        # but the sources should NOT include IP docs.
        for source in query_resp_leak.json()["sources"]:
            assert "Software_License" not in source
        
        print("[PASS] Tenant Isolation Logic Verified: Lawyer 1 cannot see Lawyer 3's data.")

if __name__ == "__main__":
    test_tenant_isolation_logic()
