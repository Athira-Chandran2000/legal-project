import os
import sys
from typing import List, Dict, Any

# Import internal modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingestion.pipeline import IngestionPipeline
from auth.database import SessionLocal
from auth.models import Lawyer

def ingest_samples():
    print("Ingesting sample legal documents for demo lawyers...")
    db = SessionLocal()
    lawyers = db.query(Lawyer).all()
    
    if not lawyers:
        print("No lawyers found. Run setup_demo.py first.")
        return

    pipeline = IngestionPipeline()

    sample_docs = {
        "NDA": [
            {
                "doc_name": "NDA_Alpha_Corp.txt",
                "text": """MUTUAL NON-DISCLOSURE AGREEMENT
                This Agreement is made between Alpha Corp and Beta Inc. 
                1. Confidential Information: Includes all trade secrets, technical data, and financial information.
                2. Non-Use: The Receiving Party shall not use Confidential Information for any purpose except to evaluate a potential business relationship.
                3. Term: This agreement expires 3 years from the effective date of May 15, 2024.
                4. Governing Law: This agreement is governed by the laws of the State of Delaware."""
            },
            {
                "doc_name": "NDA_Gamma_Services.txt",
                "text": """CONFIDENTIALITY AGREEMENT
                Gamma Services agrees to keep all client data strictly confidential.
                1. Exclusions: Confidential Information does not include information that is already public.
                2. Termination: Obligations continue for 5 years after the project ends.
                3. Jurisdiction: Any disputes will be settled in the courts of New York."""
            }
        ],
        "IP Licensing": [
            {
                "doc_name": "Software_License_Agreement.txt",
                "text": """SOFTWARE LICENSE AND DISTRIBUTION AGREEMENT
                1. Grant of License: Licensor grants Licensee a non-exclusive, non-transferable license to use the software.
                2. Royalties: Licensee shall pay a royalty of 5% of gross sales.
                3. Audit Rights: Licensor has the right to audit Licensee's records once per year.
                4. Patent Indemnity: Licensor shall defend Licensee against any patent infringement claims."""
            }
        ],
        "Real Estate": [
            {
                "doc_name": "Commercial_Lease.txt",
                "text": """COMMERCIAL LEASE AGREEMENT
                1. Premises: The retail space located at 123 Main St, Suite 400.
                2. Rent: $5,000 per month, due on the first of each month.
                3. Security Deposit: $10,000 to be held by the Landlord.
                4. Maintenance: Tenant is responsible for all interior repairs."""
            }
        ]
    }

    for lawyer in lawyers:
        specialty = lawyer.specialty
        if specialty in sample_docs:
            print(f"Indexing {len(sample_docs[specialty])} documents for {lawyer.username} ({specialty})...")
            pipeline.process_lawyer_documents(lawyer.id, sample_docs[specialty])
        else:
            print(f"No sample documents for specialty {specialty} (Lawyer: {lawyer.username})")

    db.close()
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_samples()
