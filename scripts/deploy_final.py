import subprocess
import os
import sys

def run_command(cmd):
    print(f"[STEP] {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error: {stderr}")
        return False
    print(stdout)
    return True

def deploy():
    print("Starting Final Production RAG Sync...")
    
    # 1. Re-Ingest everything with aligned IDs
    if not run_command("python scripts/ingest_samples.py"):
        print("Ingestion failed.")
        return

    # 2. Run Benchmarks
    print("[STEP] Running RAGAS Benchmarks...")
    if not run_command("python scripts/evaluate.py"):
        print("Evaluation failed.")
        return

    # 3. Push to Hugging Face
    print("Finalizing Git push to Hugging Face...")
    run_command("git add .")
    run_command('git commit -m "Final Optimized RAG Sync"')
    run_command("git push")
    
    print("Deployment and UI Sync Complete.")

if __name__ == "__main__":
    deploy()
