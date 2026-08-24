"""
shiva.py
==========
Voice pipeline: answers, acts, and sees.

    "move forward"      -> wheels turn, returns immediately
    "what's the weather" -> answered while still moving
    "what is this?"      -> camera + VLM describes it
    "stop"               -> wheels stop

ACTION does not block. Motor commands set GPIO pins and return
in microseconds, so the next question can be heard and answered
while the robot is still driving. Movement costs no CPU once
the pins are set.

ROUTES
------
    STATIC          model weights
    KNOWLEDGE_BASE  ChromaDB
    LIVE_DATA       MCP tools (news, weather)
    ACTION          drive.py functions
    VISION          camera + vision-language model

GATE ORDER
----------
Classification now runs BEFORE the A0 safety gate, so ACTION
can skip it entirely. Two reasons: "move forward" has no
content to be unsafe about, and A0 is a paid API call on every
utterance - which adds up, and adds ~400ms to a command where
latency actually matters.

The tradeoff, stated plainly: an unsafe question now reaches
the classifier before the safety check. That is a small
exposure - DistilBERT only produces a label, it does not
generate or retrieve - but it is a real change from the
original design, where nothing touched the question until A0
had cleared it.

VISION still runs A0. That is the route where a child could
ask about something they should not, and 400ms is noise next
to 30+ seconds of inference.

REMOTE INFERENCE
----------------
Ollama host is configurable. Vision on a laptop and everything
else local:

    VISION_HOST=http://192.168.1.140:11434 python3 shiva.py

Unset, everything runs on the Pi - which is the version that
ships, since a robot needing a laptop on the same WiFi is not
an offline robot.

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
import asyncio
import subprocess
import wave

import numpy as np
import requests
import chromadb
import torch
from faster_whisper import WhisperModel
from fuzzywuzzy import fuzz
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import drive
import vision
from mcp_client_integration import get_live_data, is_weather_query
from gate_a0_safety import check_child_safety
from t1_logger import log_interaction

ROBOT_NAME = "shiva"

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

#MODEL_NAME = "hf.co/puneetsiet2005/robotai-v7-grpo"
MODEL_NAME = "qwen2.5vl:7b"
EMBED_MODEL = "mxbai-embed-large"
CLASSIFIER_PATH = os.path.abspath("./distilbert_classifier_final")
PIPER_VOICE = "en_US-ryan-low.onnx"

AUDIO_DEV = "plughw:CARD=Device,DEV=0"
CACHE_THRESHOLD = 97

# Whisper mishears "robo" often enough that an exact match would
# miss it - fuzzy partial match is more forgiving.
WAKE_WORD = "robo"
WAKE_THRESHOLD = 75

# Low margin means the classifier was nearly split. On ACTION
# that matters more than elsewhere - a wrong guess moves a
# physical robot rather than returning a wrong fact.
ACTION_MIN_MARGIN = 0.30

response_cache = {}

print("loading classifier...")
tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_PATH)
classifier = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_PATH)
classifier.eval()
ID2LABEL = classifier.config.id2label

print("loading whisper...")
stt = WhisperModel("base.en", device="cpu", compute_type="int8")

rag_client = chromadb.PersistentClient(path="./robot_chroma_db_mxbai")
rag = rag_client.get_or_create_collection(name="kids_knowledge_mxbai")

try:
    from adafruit_servokit import ServoKit
    kit = ServoKit(channels=16)
except Exception as e:
    print(f"no servos ({e}) - head will stay still")
    kit = None


# ============================================================
# Speech
# ============================================================
def speak(text):
    """Text goes to piper's stdin directly. Passing it through a
    shell breaks on apostrophes and exclamation marks - the
    answer gets cut off mid-sentence at the first quote."""
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
    piper.stdin.write(clean.encode())
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
            check=True, stderr=subprocess.DEVNULL,
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
    if float(np.abs(audio).mean()) < 0.004:
        return ""

    audio = audio[::3]          # 48000 -> 16000 for whisper
    segments, _ = stt.transcribe(audio, language="en", beam_size=1)
    return " ".join(s.text for s in segments).strip()


def heard_wake_word(text):
    return fuzz.partial_ratio(WAKE_WORD, text.lower()) >= WAKE_THRESHOLD


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
    probs = torch.softmax(logits, dim=1)[0]
    top2 = torch.topk(probs, 2).values
    return float(top2[0] - top2[1])


# ============================================================
# ACTION - keyword match, non-blocking
# ============================================================
# Order matters. "stop" is checked first so it cannot be missed,
# and "come back" matches 'back' before it reaches 'come'.
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
    r = requests.post(f"{OLLAMA_HOST}/api/embeddings",
                      json={"model": EMBED_MODEL, "prompt": text})
    return r.json()["embedding"]


def retrieve(query, n=2, max_distance=0.6):
    results = rag.query(query_embeddings=[embed(query)], n_results=n)
    return [d for d, dist in zip(results["documents"][0], results["distances"][0])
            if dist < max_distance]


def generate(prompt):
    r = requests.post(f"{OLLAMA_HOST}/api/generate",
                      json={"model": MODEL_NAME, "prompt": prompt,
                            "stream": False,
                            "options": {"num_predict": 100, "temperature": 0.7}},
                      timeout=120)
    return r.json().get("response", "").strip()


FALLBACK = "Hmm, I'm not sure about that one. Ask me something else!"


# ============================================================
# Main handler
# ============================================================
def handle(question):
    route, logits = classify(question)
    conf = margin(logits)
    print(f"  [{route}] margin={conf:.2f}")

    # ACTION skips A0 - no content to police, and the API call
    # costs money and ~400ms on a command where latency matters.
    if route == "ACTION":
        if conf < ACTION_MIN_MARGIN:
            speak("I'm not sure what you want me to do.")
            return
        reply = do_action(question)
        speak(reply or "I didn't catch that command.")
        log_interaction(question, [], reply or "", "ACTION")
        return

    # Everything else goes through the safety gate.
    safe, category, redirect = check_child_safety(question, verbose=False)
    if not safe:
        print(f"  [GATE-A0] {category}")
        speak(redirect)
        log_interaction(question, [], redirect, f"GATED-A0:{category}")
        return

    if route == "VISION":
        answer = vision.look_and_describe(question, speak=speak, kit=kit)
        print(f"  answer: {answer}")
        speak(answer)
        log_interaction(question, [vision.RESIZED_PATH], answer, "VISION")
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
        else:
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

def strip_wake_word(text):
    return re.sub(rf"\b{WAKE_WORD}\w*\b[,\s]*", "", text, flags=re.I).strip()


def main():
    print(f'\n{ROBOT_NAME.title()} is listening for "{WAKE_WORD}". Ctrl+C to quit.')
    print(f"ollama: {OLLAMA_HOST}")
    print(f"vision: {vision.MODEL} @ {vision.OLLAMA_HOST}\n")
    drive.stop()

    try:
        while True:
            heard = listen()
            if not heard:
                continue

            if not heard_wake_word(heard):
                continue

            question = strip_wake_word(heard)

            # "Robo, look up" carries its own question. Only ask
            # back when the wake word arrived alone.
            if len(question.split()) < 2:
                speak("Yes?")
                question = listen()
                if not question:
                    continue

            print(f"heard: {question}")
            handle(question)

    except KeyboardInterrupt:
        pass
    finally:
        drive.stop()
        print("\nstopped")


if __name__ == "__main__":
    main()
