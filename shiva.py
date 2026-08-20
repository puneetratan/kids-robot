"""
shiva.py
==========
Voice pipeline with movement. Wake word: "Shiva".

    "Shiva, move forward"        -> wheels turn, returns immediately
    "Shiva, what's the weather"  -> answered while still moving
    "Shiva, stop"                -> wheels stop

The point of this version is that ACTION does not block. Motor
commands set GPIO pins and return in microseconds, so the next
question can be heard and answered while the robot is still
driving. Movement costs no CPU once the pins are set.

ROUTES
------
    STATIC          model weights
    KNOWLEDGE_BASE  ChromaDB
    LIVE_DATA       MCP tools (news, weather)
    ACTION          drive.py functions          <- new
    VISION          placeholder until the camera is mounted

REQUIRES
--------
    MCP server running in another terminal
    Ollama running
    ANTHROPIC_API_KEY set (for Gate A0)

Run:
    python3 shiva.py
"""

import os
import re
import time
import queue
import asyncio
import threading

import numpy as np
import requests
#import sounddevice as sd
import chromadb
import torch
from faster_whisper import WhisperModel
from fuzzywuzzy import fuzz
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import subprocess
import wave

import drive
from mcp_client_integration import get_live_data, is_weather_query
from gate_a0_safety import check_child_safety
from t1_logger import log_interaction

ROBOT_NAME = "shiva"
WAKE_WORDS = (ROBOT_NAME, "siva", "shiv", "sheva")

MODEL_NAME = "hf.co/puneetsiet2005/robotai-v7-grpo"
EMBED_MODEL = "mxbai-embed-large"
CLASSIFIER_PATH = os.path.abspath("./distilbert_classifier_final")
PIPER_VOICE = "en_US-ryan-low.onnx"

SAMPLE_RATE = 16000
DEVICE_RATE = 48000
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 1.2
CACHE_THRESHOLD = 97

response_cache = {}

AUDIO_DEV = "plughw:CARD=Device,DEV=0"
#sd.default.device = (0, 0)

print("loading classifier...")
tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_PATH)
classifier = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_PATH)
classifier.eval()
ID2LABEL = classifier.config.id2label

print("loading whisper...")
stt = WhisperModel("base.en", device="cpu", compute_type="int8")

rag_client = chromadb.PersistentClient(path="./robot_chroma_db_mxbai")
rag = rag_client.get_or_create_collection(name="kids_knowledge_mxbai")


# ============================================================
# Speech
# ============================================================
def speak(text):
    clean = re.sub(r"[^\w\s.,!?'-]", "", text)
    piper = subprocess.Popen(
        ["./piper/piper", "--model", PIPER_VOICE,
         "--length_scale", "1.3", "--output_raw"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    aplay = subprocess.Popen(
        ["aplay", "-D", AUDIO_DEV, "-r", "22050",
         "-f", "S16_LE", "-t", "raw", "-"],
        stdin=piper.stdout, stderr=subprocess.DEVNULL,
    )
    piper.stdout.close()
    piper.stdin.write(text.encode())
    piper.stdin.close()
    aplay.wait()
    time.sleep(0.3)
def listen(seconds=5):
    """Fixed-window capture via arecord. PortAudio cannot open
    this device, so sounddevice is not used."""
    path = "/tmp/heard.wav" 
    try:
        subprocess.run(
            ["arecord", "-D", AUDIO_DEV, "-f", "S16_LE",
             "-r", "48000", "-c", "1", "-d", str(seconds), "-q", path],
            check=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        time.sleep(0.5)
        return ""

    try:
        with wave.open(path, "rb") as w:
            raw = w.readframes(w.getnframes())
    except Exception:
        return ""
    if not raw:
        return ""

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    # skip near-silence - avoids running whisper on an empty room
    if float(np.abs(audio).mean()) < 0.004:
        return ""

    audio = audio[::3]          # 48000 -> 16000 for whisper
    segments, _ = stt.transcribe(audio, language="en", beam_size=1)
    return " ".join(s.text for s in segments).strip()

def heard_wake_word(text):
    lowered = text.lower()
    return any(w in lowered for w in WAKE_WORDS)


def strip_wake_word(text):
    out = text
    for w in WAKE_WORDS:
        out = re.sub(rf"\b{w}\b[,\s]*", "", out, flags=re.I)
    return out.strip()


# ============================================================
# Classification
# ============================================================
def classify(question):
    inputs = tokenizer(question, return_tensors="pt", truncation=True,
                       padding=True, max_length=64)
    with torch.no_grad():
        logits = classifier(**inputs).logits
    idx = int(torch.argmax(logits, dim=1))
    return ID2LABEL[idx], logits


def margin(logits):
    """Top1 - top2 softmax gap. Low margin on an ACTION means
    we are unsure whether this was even a command - and a wrong
    guess moves a physical robot."""
    probs = torch.softmax(logits, dim=1)[0]
    top2 = torch.topk(probs, 2).values
    return float(top2[0] - top2[1])


# ============================================================
# ACTION - keyword match, non-blocking
# ============================================================
COMMANDS = [
    (("stop", "halt", "wait", "freeze", "stay", "don't move", "enough"),
     drive.stop, "stopping"),
    (("back", "reverse", "backward", "backwards", "return"),
     drive.backward, "going back"),
    (("left",), drive.turn_left, "turning left"),
    (("right",), drive.turn_right, "turning right"),
    (("forward", "ahead", "straight", "go", "move", "drive", "come", "start"),
     drive.forward, "moving forward"),
]


def do_action(question):
    """Set pins, return immediately. The wheels keep turning
    until the next command - nothing here waits."""
    lowered = question.lower()
    for keywords, fn, reply in COMMANDS:
        if any(k in lowered for k in keywords):
            fn()
            return reply
    return None


# ============================================================
# Retrieval + generation
# ============================================================
def embed(text):
    r = requests.post("http://localhost:11434/api/embeddings",
                      json={"model": EMBED_MODEL, "prompt": text})
    return r.json()["embedding"]


def retrieve(query, n=2, max_distance=0.6):
    results = rag.query(query_embeddings=[embed(query)], n_results=n)
    return [d for d, dist in zip(results["documents"][0], results["distances"][0])
            if dist < max_distance]


def generate(prompt):
    r = requests.post("http://localhost:11434/api/generate",
                      json={"model": MODEL_NAME, "prompt": prompt,
                            "stream": False,
                            "options": {"num_predict": 100, "temperature": 0.7}},
                      timeout=120)
    return r.json().get("response", "").strip()


FALLBACK = "Hmm, I'm not sure about that one. Ask me something else!"
NO_EYES = "I can't see yet - my camera isn't installed. Ask me something else!"


# ============================================================
# Main handler
# ============================================================
def handle(question):
    safe, category, redirect = check_child_safety(question, verbose=False)
    if not safe:
        speak(redirect)
        log_interaction(question, [], redirect, f"GATED-A0:{category}")
        return

    route, logits = classify(question)
    conf = margin(logits)
    print(f"  [{route}] margin={conf:.2f}")

    if route == "ACTION":
        if conf < 0.30:
            speak("I'm not sure what you want me to do.")
            return
        reply = do_action(question)
        speak(reply or "I didn't catch that command.")
        log_interaction(question, [], reply or "", "ACTION")
        return

    if route == "VISION":
        speak(NO_EYES)
        print(f"  answer: {NO_EYES}")
        log_interaction(question, [], NO_EYES, "VISION")
        return

    for cached_q, cached_a in response_cache.items():
        if fuzz.ratio(question.lower(), cached_q) >= CACHE_THRESHOLD:
            speak(cached_a)
            return

    contexts = []
    if route == "LIVE_DATA":
        tool = "weather" if is_weather_query(question) else "news"
        print(f"  fetching via MCP ({tool})...")
        try:
            live = asyncio.run(get_live_data(question))
        except Exception as e:
            print(f"  [MCP error] {e}")
            live = None
        if live:
            contexts = [live]
    elif route == "KNOWLEDGE_BASE":
        contexts = retrieve(question)

    if route == "STATIC":
        prompt = (f"Answer this question in a fun, kid-friendly way with an "
                  f"analogy.\n\nQuestion: {question}\n\nAnswer in 2-3 sentences:")
    else:
        if contexts:
         print(f"  context: {contexts[0][:200]}")
        if not contexts:
            print(f"  answer: {FALLBACK}")
            speak(FALLBACK)
            log_interaction(question, [], FALLBACK, route)
            return
        context_text = "\n\n".join(contexts)
        prompt = (f"Use ONLY the following information to answer.\n\n"
                  f"Context:\n{context_text}\n\nQuestion: {question}\n\n"
                  f"Answer in 2-3 sentences, kid-friendly:")

    answer = generate(prompt)
    if not answer:
        answer = FALLBACK
    else:
        response_cache[question.lower()] = answer
    print(f"  answer: {answer}")
    speak(answer)
    log_interaction(question, contexts, answer, route)


def main():
    print(f"\n{ROBOT_NAME.title()} is listening. Ctrl+C to quit.\n")
    drive.stop()

    try:
        while True:
            heard = listen()
            if not heard:
                continue
            print(f"heard: {heard}")
            handle(heard)

    except KeyboardInterrupt:
        pass
    finally:
        drive.stop()
        print("\nstopped")

if __name__ == "__main__":
    main()
