# Stage 13: Hugging Face Spaces Deployment
# Build Trigger: Production Dashboard Sync v2
FROM python:3.11-slim

# Create a non-root user for HF Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install system dependencies (must be done as root)
USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*
USER user

# Install Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Pre-download models to cache in the image
RUN python -c "import torch; from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -c "from flashrank import Ranker; Ranker(model_name='ms-marco-MiniLM-L-12-v2')"

# Copy application code
COPY --chown=user . .

# Ensure data directories are writable
RUN mkdir -p data/tenants data/central_auth

# Expose port (HF Spaces default)
EXPOSE 7860

# Run the application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
