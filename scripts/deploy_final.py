import os
import sys
import json
import subprocess
import re

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_step(name, command):
    print(f"\n[STEP] {name}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"ERROR in {name}: {e.stderr}")
        return None

def update_ui_metrics(metrics):
    print("\nUpdating UI Dashboard with REAL metrics...")
    index_path = "app/static/index.html"
    with open(index_path, 'r', encoding='utf-8') as f:
        html = f.read()

    # Update Groundedness
    html = re.sub(r'id="metric-groundedness".*?>[\d\.]+<', f'id="metric-groundedness" style="font-size: 1rem;">{metrics.get("groundedness", "0.00")}<', html)
    # Update Context Rel
    html = re.sub(r'id="metric-context-rel".*?>[\d\.]+<', f'id="metric-context-rel" style="font-size: 1rem;">{metrics.get("context_rel", "0.00")}<', html)
    # Update Answer Rel
    html = re.sub(r'id="metric-answer-rel".*?>[\d\.]+<', f'id="metric-answer-rel" style="font-size: 1rem;">{metrics.get("answer_rel", "0.00")}<', html)
    # Update Precision
    html = re.sub(r'id="metric-precision".*?>[\d\.]+<', f'id="metric-precision" style="font-size: 1rem;">{metrics.get("precision", "0.00")}<', html)
    # Update Recall
    html = re.sub(r'id="metric-recall".*?>[\d\.]+<', f'id="metric-recall" style="font-size: 1rem;">{metrics.get("recall", "0.00")}<', html)
    # Update MRR
    html = re.sub(r'id="metric-mrr".*?>[\d\.]+<', f'id="metric-mrr" style="font-size: 1rem;">{metrics.get("mrr", "0.00")}<', html)

    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print("UI Dashboard synchronized.")

def main():
    print("--- LEXGUARD FINAL DEPLOYMENT & SYNC ---")
    
    # 1. Re-index
    run_step("Re-indexing documents", "python scripts/setup_demo.py && python scripts/ingest_samples.py")
    
    # 2. Run Evaluation
    output = run_step("Running RAGAS Benchmarks", "python scripts/evaluate.py")
    
    if output:
        # Parse output for metrics
        metrics = {}
        try:
            # Extract RAGAS metrics
            metrics["groundedness"] = re.search(r"Groundedness \(Faithfulness\): ([\d\.]+)", output).group(1)
            metrics["answer_rel"] = re.search(r"Answer Relevance: ([\d\.]+)", output).group(1)
            metrics["context_rel"] = re.search(r"Context Relevance: ([\d\.]+)", output).group(1)
            
            # Extract Retrieval metrics
            metrics["precision"] = re.search(r"Precision@5: ([\d\.]+)", output).group(1)
            metrics["recall"] = re.search(r"Recall@5:\s+([\d\.]+)", output).group(1)
            metrics["mrr"] = re.search(r"MRR:\s+([\d\.]+)", output).group(1)
            
            update_ui_metrics(metrics)
        except Exception as e:
            print(f"Could not parse metrics from output: {e}")

    # 3. Final Push
    print("\nFinalizing Git push to Hugging Face...")
    run_step("Pushing to HF", 'git add . && git commit -m "Final Optimized RAG Sync" && git push hf main --force')

if __name__ == "__main__":
    main()
