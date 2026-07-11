from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time

MODEL_PATH = "./distilbert_classifier_final"

print("Loading model...")
load_start = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
load_time = time.time() - load_start
print(f"Model loaded in {load_time:.2f}s\n")

id2label = {0: "STATIC", 1: "KNOWLEDGE_BASE", 2: "LIVE_DATA"}

def classify_query(question):
    start = time.time()
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_id = torch.argmax(outputs.logits, dim=1).item()
    elapsed = time.time() - start
    return id2label[predicted_id], elapsed

test_questions = [
    "What is gravity?",
    "Tell me about Ramayana",
    "Who is hosting the FIFA World Cup in 2026?",
    "Is the Earth still spinning?",
]

print("Running classification (with per-question timing):\n")
for q in test_questions:
    category, elapsed = classify_query(q)
    print(f"Q: {q}")
    print(f"  → {category}  ({elapsed*1000:.1f}ms)\n")
