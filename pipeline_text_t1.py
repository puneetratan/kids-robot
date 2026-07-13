"""
pipeline_text_t1.py
(pipeline_text.py + Tier 1 live evaluation logging)
=================
Same routing logic as voice_pipeline_routed.py but text in, text out.
No Whisper, no Piper, no sounddevice — just type and get an answer.
Useful for testing classifier routing and MCP tools without mic/speaker setup.

Usage:
    python3 pipeline_text.py
"""

import requests
import json
import os
import time
import asyncio
from fuzzywuzzy import fuzz
import chromadb
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from mcp_client_integration import get_live_data, is_weather_query
from t1_logger import log_interaction

# ============================================================
# Query Classifier setup
# ============================================================
CLASSIFIER_PATH = os.path.abspath("./distilbert_classifier_final")
classifier_tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_PATH)
classifier_model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_PATH)
classifier_model.eval()
ID2LABEL = {0: "STATIC", 1: "KNOWLEDGE_BASE", 2: "LIVE_DATA"}

_ = classifier_model(**classifier_tokenizer("warmup", return_tensors="pt"))

# ============================================================
# RAG setup
# ============================================================
rag_client = chromadb.PersistentClient(path="./robot_chroma_db_mxbai")
rag_collection = rag_client.get_or_create_collection(name="kids_knowledge_mxbai")

CACHE_THRESHOLD = 97
response_cache = {}

MODEL_NAME = "hf.co/puneetsiet2005/robotai-v7-grpo"
EMBED_MODEL = "mxbai-embed-large"


def classify_query(question):
    inputs = classifier_tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = classifier_model(**inputs)
    predicted_id = torch.argmax(outputs.logits, dim=1).item()
    return ID2LABEL[predicted_id]


def get_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text}
    )
    return response.json()["embedding"]


def retrieve_context(query, n_results=2, max_distance=0.6):
    query_embedding = get_embedding(query)
    results = rag_collection.query(query_embeddings=[query_embedding], n_results=n_results)

    good_docs = []
    for doc, dist, meta in zip(results['documents'][0], results['distances'][0], results['metadatas'][0]):
        if dist < max_distance:
            good_docs.append(doc)
    return good_docs


def retrieve_live_data(query):
    try:
        return asyncio.run(get_live_data(query))
    except Exception as e:
        print(f"[MCP ERROR] {e}")
        return None


def ask(question):
    question_lower = question.lower()

    # Fuzzy cache check
    for cached_q, cached_ans in response_cache.items():
        score = fuzz.ratio(question_lower, cached_q)
        if score >= CACHE_THRESHOLD:
            print(f"[CACHE HIT]\n")
            print(f"Robot: {cached_ans}\n")
            return

    category = classify_query(question)
    logged_contexts = []

    if category == "STATIC":
        print(f"[STATIC] Answering from model knowledge...")
        prompt = f"""Answer this question in a fun, kid-friendly way with emojis and an analogy.

Question: {question}

Answer in 2-3 sentences:"""

    elif category == "LIVE_DATA":
        tool = "WEATHER" if is_weather_query(question) else "NEWS"
        print(f"[LIVE_DATA → {tool}] Fetching via MCP...")
        live_context = retrieve_live_data(question)
        if live_context:
            logged_contexts = [live_context]

        if live_context:
            prompt = f"""Use ONLY the following live information to answer the question. Mention all key facts from the context, not just one detail.

Context:
{live_context}

Question: {question}

Answer in 2-3 sentences, kid-friendly with emojis:"""
        else:
            prompt = None

    else:
        print(f"[KNOWLEDGE_BASE] Retrieving from knowledge base...")
        context_docs = retrieve_context(question)
        logged_contexts = context_docs

        if context_docs:
            context = "\n\n".join(context_docs)
            prompt = f"""Use ONLY the following information to answer the question. Mention all key facts from the context, not just one detail.

Context:
{context}

Question: {question}

Answer in 2-3 sentences, kid-friendly with emojis:"""
        else:
            prompt = None

    if prompt is None:
        fallback = "Hmm, I don't have information about that yet! Ask me something else, or teach me about it! 🤖📚"
        print(f"Robot: {fallback}\n")
        log_interaction(question, logged_contexts, fallback, category)
        return

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": True,
                "options": {"num_predict": 100, "temperature": 0.7}
            },
            stream=True,
            timeout=120
        )

        print("Robot: ", end="", flush=True)
        full_response = ""

        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("response", "")
                full_response += token
                print(token, end="", flush=True)
                if chunk.get("done"):
                    break

        print("\n")

        if full_response and len(full_response) > 5:
            response_cache[question_lower] = full_response

        log_interaction(question, logged_contexts, full_response, category)

    except Exception as e:
        print(f"\n[ERROR] {e}\n")


def main():
    print("Robot Text Pipeline Ready!")
    print(f"Knowledge base: {rag_collection.count()} documents")
    print("Type your question and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            ask(question)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
