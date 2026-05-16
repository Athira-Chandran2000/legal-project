import json
import os
import random
from datasets import load_dataset

def prepare_eval_data():
    print("Preparing high-quality 200-question evaluation set from CUAD...")
    
    try:
        print("Streaming CUAD Q&A pairs from Hugging Face Hub...")
        dataset = load_dataset("atticus_legal/cuad", split="test") # Use test split for eval
        all_qas = []
        for row in dataset:
            question = row.get("question", "")
            # In CUAD Hub, answers are often in a list of texts
            answers = row.get("answers", {}).get("text", [])
            if not answers:
                 # Try alternative schema
                 answers = row.get("answers", [])
            
            if answers and len(str(answers[0])) > 10:
                all_qas.append({
                    "question": question,
                    "expected_answer": str(answers[0]),
                    "doc_name": row.get("id", "CUAD_Doc")
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
        
    # 2. Isolation Test Set (100 Questions)
    # We'll tag these as "security_check"
    # (In actual testing, these would be asked to lawyers who DON'T have these docs)
    
    output_path = "data/eval_set.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(eval_set, f, indent=2)
        
    print(f"Evaluation set prepared with {len(eval_set)} questions. Saved to {output_path}")

if __name__ == "__main__":
    prepare_eval_data()
