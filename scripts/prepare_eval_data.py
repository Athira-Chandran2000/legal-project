import json
import os
import random
from datasets import load_dataset

def prepare_eval_data():
    print("Preparing high-quality 200-question evaluation set from CUAD...")
    
    try:
        print("Streaming official CUAD Q&A pairs from the Hub (chenghao/cuad_qa)...")
        dataset = load_dataset("chenghao/cuad_qa", split="test")
        all_qas = []
        for row in dataset:
            question = row.get("question", "")
            answers = row.get("answers", {}).get("text", [])
            # Use the exact same ID field as ingestion
            doc_id = row.get("id", "CUAD_Doc")
            
            if answers and len(str(answers[0])) > 10:
                all_qas.append({
                    "question": question,
                    "expected_answer": str(answers[0]),
                    "doc_name": doc_id
                })
        print(f"Total candidate answerable questions: {len(all_qas)}")
    except Exception as e:
        print(f"Error loading Hub dataset: {e}")
        return

    # Sample 200 main questions
    if len(all_qas) >= 200:
        eval_set = random.sample(all_qas, 200)
    else:
        eval_set = all_qas
        
    output_path = "data/eval_set.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(eval_set, f, indent=2)
        
    print(f"Evaluation set prepared with {len(eval_set)} questions. Saved to {output_path}")

if __name__ == "__main__":
    prepare_eval_data()
