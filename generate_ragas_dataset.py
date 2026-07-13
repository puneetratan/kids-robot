"""
generate_ragas_dataset.py
===========================
Step 2 of Video 11: build the 50-question RAGAS eval dataset.

What it does:
  1. Claude generates 50 questions + ground truths
     (30 ChromaDB-oriented, 20 MCP-oriented per our split)
  2. Each question runs through the REAL pipeline:
     DistilBERT classify -> ChromaDB retrieve OR MCP fetch -> Llama answer
  3. Saves everything RAGAS needs to ragas_dataset.json:
     { question, contexts, answer, ground_truth, category, source }

Run on the Pi (MCP server must be running first):
    export ANTHROPIC_API_KEY=your_key
    python3 generate_ragas_dataset.py

NOTE on ground truths: Claude writes them, but REVIEW THEM MANUALLY
before running the eval — especially the 10 news questions, where
Claude's knowledge may be stale. Wrong ground truth = meaningless
context_recall scores. Editing ragas_dataset.json by hand is expected.
"""

import os
import json
import time
import asyncio
import requests
import anthropic
import chromadb
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from mcp_client_integration import get_live_data, is_weather_query

# ============================================================
# Claude (question + ground truth generation)
# ============================================================
client = anthropic.Anthropic()
GEN_MODEL = "claude-sonnet-4-5"

# ============================================================
# Pipeline components (same as pipeline_text.py)
# ============================================================
print("Loading classifier...")
CLASSIFIER_PATH = os.path.abspath("./distilbert_classifier_final")
classifier_tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_PATH)
classifier_model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_PATH)
classifier_model.eval()
ID2LABEL = {0: "STATIC", 1: "KNOWLEDGE_BASE", 2: "LIVE_DATA"}
_ = classifier_model(**classifier_tokenizer("warmup", return_tensors="pt"))

print("Loading ChromaDB...")
rag_client = chromadb.PersistentClient(path="./robot_chroma_db_mxbai")
rag_collection = rag_client.get_or_create_collection(name="kids_knowledge_mxbai")
print(f"Knowledge base: {rag_collection.count()} documents")

MODEL_NAME = "hf.co/puneetsiet2005/robotai-v7-grpo"
EMBED_MODEL = "mxbai-embed-large"


def classify_query(question):
    inputs = classifier_tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = classifier_model(**inputs)
    return ID2LABEL[torch.argmax(outputs.logits, dim=1).item()]


def get_embedding(text):
    r = requests.post("http://localhost:11434/api/embeddings",
                      json={"model": EMBED_MODEL, "prompt": text})
    data = r.json()
    if "embedding" not in data:
        raise RuntimeError(f"Ollama embeddings error: {data}")
    return data["embedding"]


def retrieve_context(query, n_results=2, max_distance=0.6):
    emb = get_embedding(query)
    results = rag_collection.query(query_embeddings=[emb], n_results=n_results)
    docs = []
    for doc, dist in zip(results["documents"][0], results["distances"][0]):
        if dist < max_distance:
            docs.append(doc)
    return docs


def generate_answer(question, context):
    if context:
        prompt = f"""Use ONLY the following information to answer the question. Mention all key facts from the context, not just one detail.

Context:
{context}

Question: {question}

Answer in 2-3 sentences, kid-friendly with emojis:"""
    else:
        prompt = f"""Answer this question in a fun, kid-friendly way with emojis and an analogy.

Question: {question}

Answer in 2-3 sentences:"""

    r = requests.post("http://localhost:11434/api/generate",
                      json={"model": MODEL_NAME, "prompt": prompt,
                            "stream": False,
                            "options": {"num_predict": 100, "temperature": 0.7}},
                      timeout=120)
    return r.json().get("response", "").strip()


# ============================================================
# Step 1: Claude generates questions + ground truths
# ============================================================
def generate_questions():
    """Ask Claude for the full 50-question set as JSON."""
    prompt = """Generate evaluation questions for a kids education robot's RAG system.
The robot has:
- A knowledge base with topics: gravity, photosynthesis, solar system, Ramayana,
  dinosaurs, ocean animals, volcanoes, the human body, math puzzles, space travel,
  World Cup 2026 (co-hosted by USA, Canada, Mexico), Mars rovers, the ISS
- Live tools: current news (Google News) and weather (Columbus, OH)

Generate EXACTLY 50 questions as a JSON array. Each item:
{"question": "...", "ground_truth": "...", "expected_source": "..."}

Distribution:
- 20 questions answerable from the knowledge base topics above
  (expected_source: "knowledge_base")
- 10 tricky knowledge-base questions: ambiguous phrasing, entity ambiguity
  (e.g. which Ronaldo?), thin-context traps, questions mixing two topics
  (expected_source: "knowledge_base")
- 10 current news questions a kid might ask (sports, space, events)
  (expected_source: "news")
- 10 weather questions phrased different ways
  (expected_source: "weather")

Ground truths: 1-2 sentences, factually correct, simple language.
For weather questions, ground_truth should be "WEATHER_LIVE" (checked at runtime).
For news questions where the answer changes daily, write the ground truth
as of your best current knowledge and mark uncertain ones with [VERIFY].

Output ONLY the JSON array. No markdown fences, no commentary."""

    response = client.messages.create(
        model=GEN_MODEL,
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    # strip accidental fences
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text.replace("json", "", 1).strip()
    return json.loads(text)


# ============================================================
# Step 2: run each question through the real pipeline
# ============================================================
def run_pipeline(question):
    category = classify_query(question)

    if category == "STATIC":
        contexts = []
        answer = generate_answer(question, None)

    elif category == "LIVE_DATA":
        try:
            live = asyncio.run(get_live_data(question))
        except Exception as e:
            live = None
            print(f"    [MCP ERROR] {e}")
        contexts = [live] if live else []
        answer = generate_answer(question, live) if live else \
            "Hmm, I don't have information about that yet!"

    else:  # KNOWLEDGE_BASE
        docs = retrieve_context(question)
        contexts = docs
        answer = generate_answer(question, "\n\n".join(docs)) if docs else \
            "Hmm, I don't have information about that yet!"

    return category, contexts, answer


def main():
    print("\nStep 1: Generating 50 questions via Claude...")
    questions = generate_questions()
    print(f"  Got {len(questions)} questions")

    dataset = []
    print("\nStep 2: Running each through the pipeline...\n")
    for i, item in enumerate(questions, 1):
        q = item["question"]
        print(f"[{i:02d}/50] {q}")
        try:
            category, contexts, answer = run_pipeline(q)
            print(f"        -> {category} | {len(contexts)} context(s)")
            dataset.append({
                "question": q,
                "contexts": contexts,
                "answer": answer,
                "ground_truth": item["ground_truth"],
                "expected_source": item.get("expected_source", ""),
                "classified_as": category,
            })
        except Exception as e:
            print(f"        -> ERROR: {e}")
            dataset.append({
                "question": q, "contexts": [], "answer": f"PIPELINE_ERROR: {e}",
                "ground_truth": item["ground_truth"],
                "expected_source": item.get("expected_source", ""),
                "classified_as": "ERROR",
            })
        time.sleep(0.5)  # be gentle to news RSS / NWS

    with open("ragas_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(dataset)} rows to ragas_dataset.json")
    print("\nNEXT: manually review ground truths (especially [VERIFY] ones),")
    print("then run: python3 run_ragas_eval.py")


if __name__ == "__main__":
    main()
