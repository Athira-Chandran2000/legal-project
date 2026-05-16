# LexGuard Deployment Guide | Hugging Face Spaces

This guide outlines the steps to deploy the LexGuard Legal RAG system to Hugging Face Spaces using Docker.

## Prerequisites
1. A Hugging Face account.
2. A Groq API Key (for LLM inference).
3. A Hugging Face Write Token (`HF_TOKEN`).

## Deployment Steps

### 1. Create a New Space
- Go to [huggingface.co/new-space](https://huggingface.co/new-space).
- **Owner**: Your username.
- **Space Name**: `legal-rag-demo`.
- **SDK**: Select **Docker**.
- **Template**: Choose `Blank`.
- **Public/Private**: As per your preference.

### 2. Configure Secrets
In your Space's **Settings** tab, add the following variables under "Variables and secrets":
- `GROQ_API_KEY`: Your Groq API key.
- `JWT_SECRET_KEY`: A random secure string (e.g., `openssl rand -hex 32`).

### 3. Deploy via Git
Run the following commands in your local terminal:
```bash
# 1. Add the HF Space as a remote
git remote add hf https://huggingface.co/spaces/<your-username>/legal-rag-demo

# 2. Add your HF Token for authentication
git config --global credential.helper store
# (You will be prompted for your token on the first push)

# 3. Force push to the main branch
git push hf main --force
```

### 4. Automated CI/CD (Optional)
The project includes a `.github/workflows/main.yml` file. To enable automated deployment on every push:
1. Go to your **GitHub Repository Settings** > **Secrets and variables** > **Actions**.
2. Add the following repository secrets:
   - `GROQ_API_KEY`: Your Groq API key.
   - `HF_TOKEN`: Your Hugging Face write token.
   - `JWT_SECRET_KEY`: Same secret used in the Space settings.

## Post-Deployment Validation
Once the build is complete, visit your Space URL.
1. Run the **"Verify Zero-Leakage"** smoke test from the sidebar.
2. Check the **"Industrial Quality Benchmarks"** to ensure the RAG pipeline is performing as expected.
3. Use the demo credentials (from `README.md`) to login and test the isolated chat.
