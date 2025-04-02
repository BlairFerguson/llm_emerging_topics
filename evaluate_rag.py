import pandas as pd
import requests
import time
import json
from tqdm import tqdm

# === CONFIGURATION ===
API_URL = "http://localhost:8000"
PARQUET_PATH = "validation-00000-of-00001.parquet"
RESULTS_PATH = "rag_evaluation_results.jsonl"

# === Step 1: Load Data ===
df = pd.read_parquet(PARQUET_PATH)
data_records = df.to_dict(orient="records")
import random
data_records = random.sample(data_records, 100)



# === Step 2: Upload Contexts (chunked) ===
def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

all_contexts = list({record["context"] for record in data_records})
chunked_contexts = list(chunk_list(all_contexts, 50))

print(f"Uploading {len(all_contexts)} unique contexts in {len(chunked_contexts)} chunks...")
for chunk in tqdm(chunked_contexts, desc="Uploading contexts"):
    upload_payload = {"texts": chunk}
    resp = requests.post(f"{API_URL}/upload", json=upload_payload)
    if resp.status_code != 200:
        print(f"Upload failed: {resp.status_code} - {resp.text}")
        break

# === Step 3: Query and Collect Results ===
results = []
for record in tqdm(data_records, desc="Querying and evaluating"):
    question = record["question"]
    reference_answer = record["answers"]["text"][0] if record["answers"]["text"] else ""
    context_uploaded = record["context"]
    question_id = record["id"]

    payload = {"new_message": {"role": "user", "content": question}}

    start_time = time.time()
    try:
        response = requests.post(f"{API_URL}/generate", json=payload)
        elapsed = time.time() - start_time

        if response.status_code == 200:
            result_data = response.json()
            system_answer = result_data.get("generated_text", "").strip()
            retrieved_contexts = result_data.get("contexts", [])

            # Simple accuracy: reference answer substring match
            match = reference_answer.lower() in system_answer.lower()

            results.append({
                "id": question_id,
                "question": question,
                "reference_answer": reference_answer,
                "system_answer": system_answer,
                "retrieved_contexts": retrieved_contexts,
                "context_uploaded": context_uploaded,
                "model_details": {
                    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "generation_model": "Qwen/Qwen2.5-0.5B-Instruct",
                    "chunking": "sentence-window",
                    "similarity_top_k": 10,
                    "temperature": 0.1
                },
                "metrics": {
                    "accuracy_match": match,
                    "time_taken": elapsed
                }
            })
        else:
            print(f"[{question_id}] Error {response.status_code}: {response.text}")

    except Exception as e:
        print(f"[{question_id}] Exception: {e}")

# === Step 4: Save Results ===
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    for entry in results:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

import csv

CSV_PATH = "rag_evaluation_results.csv"

# === Step 5: Save to CSV ===
csv_fields = [
    "id", "question", "reference_answer", "system_answer", "accuracy_match", "time_taken",
    "embedding_model", "generation_model", "chunking", "similarity_top_k", "temperature"
]

with open(CSV_PATH, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
    writer.writeheader()
    for r in results:
        writer.writerow({
            "id": r["id"],
            "question": r["question"],
            "reference_answer": r["reference_answer"],
            "system_answer": r["system_answer"],
            "accuracy_match": r["metrics"]["accuracy_match"],
            "time_taken": round(r["metrics"]["time_taken"], 4),
            "embedding_model": r["model_details"]["embedding_model"],
            "generation_model": r["model_details"]["generation_model"],
            "chunking": r["model_details"]["chunking"],
            "similarity_top_k": r["model_details"]["similarity_top_k"],
            "temperature": r["model_details"]["temperature"]
        })

# === Step 5: Summary ===
total = len(results)
correct = sum(r["metrics"]["accuracy_match"] for r in results)
avg_time = sum(r["metrics"]["time_taken"] for r in results) / total if total else 0

print("\n✅ RAG Evaluation Complete")
print(f"Total questions evaluated: {total}")
print(f"Correct matches: {correct}")
print(f"Accuracy: {correct / total:.2%}")
print(f"Average response time: {avg_time:.2f} seconds")
