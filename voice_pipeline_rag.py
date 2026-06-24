"""
voice_pipeline_rag.py
======================
RAG-enabled version of the robot voice pipeline.

Difference from voice_pipeline.py:
- Before answering, retrieves relevant context from ChromaDB knowledge base
- If relevant context found -> answer is grounded in retrieved facts
- If NO relevant context found -> robot honestly says "I don't know yet"
  instead of letting the LLM guess from its own (sometimes wrong) training knowledge

Everything else (recording, transcription, TTS, caching) stays identical
to the original voice_pipeline.py so the RAG difference is isolated and clear.
"""

import sounddevice as sd
import numpy as np
from scipy import signal
from faster_whisper import WhisperModel
import requests
import json
import subprocess
from fuzzywuzzy import fuzz
import chromadb

# ============================================================
# Audio device setup
# FIX: device 0 = USB Composite Device (the actual mic!)
# Run `python3 -c "import sounddevice as sd; print(sd.query_devices())"`
# after any reboot/OS change - device numbers can shift!
# ============================================================
sd.default.device = (0, 0)
sd.default.samplerate = 48000

# ============================================================
# Whisper STT setup (same as original)
# ============================================================
print("Loading Whisper model...")
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
print("Whisper ready!")

# ============================================================
# RAG setup - pointing to mxbai-embed-large database
# (nomic-embed-text version had the Ramayana retrieval problem!)
# ============================================================
print("Loading knowledge base...")
rag_client = chromadb.PersistentClient(path="./robot_chroma_db_mxbai")
rag_collection = rag_client.get_or_create_collection(name="kids_knowledge_mxbai")
print(f"Knowledge base ready! {rag_collection.count()} documents loaded.")

CACHE_THRESHOLD = 97
response_cache = {}

MODEL_NAME = "hf.co/puneetsiet2005/robotai-v7-grpo"
EMBED_MODEL = "mxbai-embed-large"


def record_audio(duration=4, samplerate=48000):
    print(f"Listening for {duration} seconds...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()


def transcribe(audio, samplerate=48000):
    print("Transcribing...")
    audio_resampled = signal.resample(audio, int(len(audio) * 16000 / samplerate))
    segments, _ = whisper_model.transcribe(audio_resampled, language="en")
    text = " ".join([s.text for s in segments])
    return text.strip()


def speak(text):
    print(f"Speaking: {text}")
    subprocess.run([
        "/home/pi/robot/piper/piper",
        "--model", "en_US-ryan-low.onnx",
        "--length_scale", "1.3",
        "--output_file", "/tmp/response.wav"
    ], input=text.encode(), check=True)
    subprocess.run(["aplay", "/tmp/response.wav"], check=True)

# ============================================================
# RAG retrieval functions
# ============================================================
def get_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text}
    )
    return response.json()["embedding"]


def retrieve_context(query, n_results=2, max_distance=0.6):
    """
    Searches the knowledge base for relevant facts.
    Returns only matches below the distance threshold -
    weak/irrelevant matches are filtered out, NOT passed to the LLM.
    """
    query_embedding = get_embedding(query)
    results = rag_collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    good_docs = []
    for doc, dist, meta in zip(
        results['documents'][0],
        results['distances'][0],
        results['metadatas'][0]
    ):
        if dist < max_distance:
            good_docs.append(doc)
            print(f"  Using [{meta['topic']}] (distance: {dist:.3f})")
        else:
            print(f"  Skipping [{meta['topic']}] (distance: {dist:.3f}) - too weak!")

    return good_docs


def ask_llama_streaming_rag(question):
    """
    Same streaming logic as the original ask_llama_streaming(),
    but now RAG-grounded: retrieves context first, and builds
    a different prompt depending on whether relevant info was found.
    """
    question_lower = question.lower()

    # Fuzzy cache check (same as original)
    for cached_q, cached_ans in response_cache.items():
        score = fuzz.ratio(question_lower, cached_q)
        if score >= CACHE_THRESHOLD:
            print(f"Cache hit! ({score}% match)")
            speak(cached_ans)
            return cached_ans, True

    print(f"Searching knowledge base...")
    context_docs = retrieve_context(question)

    if context_docs:
        context = "\n\n".join(context_docs)
        prompt = f"""Use ONLY the following information to answer the question. Mention all key facts from the context, not just one detail.

Context:
{context}

Question: {question}

Answer in 2-3 sentences, kid-friendly with emojis:"""
    else:
        # Honest fallback instead of letting the model guess
        prompt = f"""You don't have specific information about this in your knowledge base.
Tell the child honestly and kindly that you don't know this yet, in a fun kid-friendly way.
Encourage them to ask something else or teach you about it.

Question: {question}

Answer in 2 sentences with emojis:"""

    print(f"Asking Llama...")
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "num_predict": 100,
                    "temperature": 0.7,
                }
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
                print(token, end="", flush=True)

                if any(buffer.rstrip().endswith(p) for p in ['.', '!', '?']):
                    speak(buffer.strip())
                    buffer = ""

                if chunk.get("done"):
                    break

        if buffer.strip():
            speak(buffer.strip())

        print()

        # Only cache RAG-grounded answers (skip caching "I don't know" responses,
        # so if the knowledge base is updated later, the robot can re-check!)
        if full_response and len(full_response) > 5 and context_docs:
            response_cache[question_lower] = full_response
            print(f"Cached!")

        return full_response, False

    except Exception as e:
        print(f"Error: {e}")
        return "Sorry, I had trouble thinking!", False


def main():
    print("RAG-Enabled Voice Robot Ready!")
    print(f"Knowledge base: {rag_collection.count()} documents")
    print("Say something! (Ctrl+C to stop)\n")

    while True:
        try:
            audio = record_audio()
            text = transcribe(audio)

            if not text:
                print("Didn't catch that, try again!")
                continue

            print(f"You said: {text}")
            response, from_cache = ask_llama_streaming_rag(text)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
