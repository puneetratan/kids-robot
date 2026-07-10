"""
voice_pipeline_routed.py
==========================
Adds intelligent query routing on top of the RAG voice pipeline.

Before answering, a fine-tuned DistilBERT classifier decides:
- STATIC        -> skip RAG entirely, answer directly from the model
- KNOWLEDGE_BASE -> retrieve from the curated ChromaDB knowledge base
- LIVE_DATA      -> routes to MCP tools (weather or news based on keywords)
"""

import sounddevice as sd
import numpy as np
from scipy import signal
from faster_whisper import WhisperModel
import requests
import json
import subprocess
import os
import time
import asyncio
from fuzzywuzzy import fuzz
import chromadb
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from mcp_client_integration import get_live_data

# ============================================================
# Audio device setup
# ============================================================
sd.default.device = (0, 0)
sd.default.samplerate = 48000

# ============================================================
# Whisper STT setup
# ============================================================
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

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


def record_audio(duration=4, samplerate=48000):
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()


def transcribe(audio, samplerate=48000):
    audio_resampled = signal.resample(audio, int(len(audio) * 16000 / samplerate))
    segments, _ = whisper_model.transcribe(audio_resampled, language="en")
    text = " ".join([s.text for s in segments])
    return text.strip()


def speak(text):
    subprocess.run([
        "/home/pi/robot/piper/piper",
        "--model", "en_US-ryan-low.onnx",
        "--length_scale", "1.3",
        "--output_file", "/tmp/response.wav"
    ], input=text.encode(), check=True, capture_output=True)
    subprocess.run(["aplay", "/tmp/response.wav"], check=True, capture_output=True)


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


def ask_llama_streaming_routed(question):
    question_lower = question.lower()

    # Fuzzy cache check
    for cached_q, cached_ans in response_cache.items():
        score = fuzz.ratio(question_lower, cached_q)
        if score >= CACHE_THRESHOLD:
            print(f"[CACHE HIT]")
            speak(cached_ans)
            return cached_ans, True

    category = classify_query(question)

    if category == "STATIC":
        print(f"[STATIC] Answering from model knowledge...")
        prompt = f"""Answer this question in a fun, kid-friendly way with emojis and an analogy.

Question: {question}

Answer in 2-3 sentences:"""

    elif category == "LIVE_DATA":
        # Keyword detection for weather vs news happens inside mcp_client_integration
        from mcp_client_integration import is_weather_query
        tool = "WEATHER" if is_weather_query(question) else "NEWS"
        print(f"[LIVE_DATA → {tool}] Fetching via MCP...")
        live_context = retrieve_live_data(question)

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
        answer = "Hmm, I don't have information about that yet! Ask me something else, or teach me about it! 🤖📚"
        speak(answer)
        return answer, False

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

        full_response = ""
        buffer = ""

        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("response", "")
                buffer += token
                full_response += token

                if any(buffer.rstrip().endswith(p) for p in ['.', '!', '?']):
                    speak(buffer.strip())
                    buffer = ""

                if chunk.get("done"):
                    break

        if buffer.strip():
            speak(buffer.strip())

        if full_response and len(full_response) > 5:
            response_cache[question_lower] = full_response

        return full_response, False

    except Exception as e:
        print(f"[ERROR] {e}")
        return "Sorry, I had trouble thinking!", False


def main():
    print("Routed Voice Robot Ready!")
    print(f"Knowledge base: {rag_collection.count()} documents")
    print("Say something! (Ctrl+C to stop)\n")

    while True:
        try:
            audio = record_audio()
            text = transcribe(audio)

            if not text:
                print("Didn't catch that, try again!")
                continue

            print(f"\nYou said: {text}")
            ask_llama_streaming_routed(text)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
