import os
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TENANT_DATA_DIR = os.path.join(BASE_DIR, "data", "tenants")

class TenantManager:
    @staticmethod
    def get_tenant_path(lawyer_id: str) -> str:
        """Returns the absolute path to a lawyer's isolated data directory."""
        path = os.path.join(TENANT_DATA_DIR, lawyer_id)
        return path

    @staticmethod
    def initialize_tenant(lawyer_id: str):
        """Creates the isolated directory structure for a new lawyer."""
        tenant_path = TenantManager.get_tenant_path(lawyer_id)
        os.makedirs(tenant_path, exist_ok=True)
        # Create subdirectories if needed (e.g., for raw docs)
        os.makedirs(os.path.join(tenant_path, "docs"), exist_ok=True)
        return tenant_path

    @staticmethod
    def get_tenant_index_paths(lawyer_id: str):
        """Returns paths to all tenant-specific index files."""
        tenant_path = TenantManager.get_tenant_path(lawyer_id)
        return {
            "faiss_index": os.path.join(tenant_path, "index.faiss"),
            "metadata_db": os.path.join(tenant_path, "metadata.db"),
            "bm25_pickle": os.path.join(tenant_path, "bm25.pkl"),
            "chunks_json": os.path.join(tenant_path, "chunks.json"),
            "docs_dir": os.path.join(tenant_path, "docs")
        }

    @staticmethod
    def delete_tenant_data(lawyer_id: str):
        """Completely removes all data for a lawyer."""
        tenant_path = TenantManager.get_tenant_path(lawyer_id)
        if os.path.exists(tenant_path):
            shutil.rmtree(tenant_path)
            return True
        return False
