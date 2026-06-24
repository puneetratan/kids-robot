import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import subprocess
import requests
import json
from scipy import signal
import os
import time
import psutil
from fuzzywuzzy import fuzz

# Change to robot directory
os.chdir('/home/pi/robot')

# Audio settings
sd.default.device = (0, 0)
sd.default.samplerate = 48000

# Cache settings
response_cache = {}
CACHE_THRESHOLD = 97

#ACTIVE_MODEL = "hf.co/puneetsiet2005/robotai-v3-bad-demo:latest"
#ACTIVE_MODEL = "robotai-v4-dpo-120-kids:latest"
ACTIVE_MODEL = "robotai-v7-grpo:latest"
# Load Whisper
print("Loading Whisper model...")
model = WhisperModel("tiny", device="cpu", compute_type="int8")
print("✅ Whisper ready!")

def warmup_llama():
    print("🔄 Warming up Llama...")
    try:
        requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": ACTIVE_MODEL,
                "prompt": "hi",
                "stream": False,
                "options": {"num_predict": 1}
            },
            timeout=120
        )
        print("✅ Llama warmed up!")
    except:
        print("⚠️ Warmup failed!")

def check_memory():
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    print(f"\n💾 Memory: {mem.percent}% used | {available_gb:.2f}GB free")
    if available_gb < 0.5:
        print("⚠️ LOW MEMORY! Restarting Ollama...")
        subprocess.run(['sudo', 'systemctl', 'restart', 'ollama'])
        time.sleep(10)

def record_audio(duration=4, samplerate=48000):
    print("\n🎤 Listening...")
    audio = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    return audio.flatten()

def transcribe(audio, samplerate=48000):
    print("🧠 Transcribing...")
    resampled = signal.resample(
        audio,
        int(len(audio) * 16000 / samplerate)
    )
    segments, _ = model.transcribe(resampled, language="en")
    return " ".join([s.text for s in segments]).strip()

def ask_llama_streaming(question):
    question_lower = question.lower().strip()
    total_tts_time = 0  # ← Add this!

    # Cache check...
    for cached_q, cached_ans in response_cache.items():
        ratio1 = fuzz.ratio(question_lower, cached_q)
        ratio2 = fuzz.partial_ratio(question_lower, cached_q)
        ratio3 = fuzz.token_sort_ratio(question_lower, cached_q)
        ratio4 = fuzz.token_set_ratio(question_lower, cached_q)
        best_score = max(ratio1, ratio2, ratio3, ratio4)

        if best_score >= CACHE_THRESHOLD:
            print(f"⚡ Cache hit! ({best_score}% match)")
            tts_start = time.time()
            speak(cached_ans)
            total_tts_time = time.time() - tts_start
            return cached_ans, True, total_tts_time  # ← Return tts time!

    print(f"💬 Asking Llama...")
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": ACTIVE_MODEL,
                "prompt": f"Answer in 2 sentences only: {question}",
                "stream": True,
                "options": {
                    "num_predict": 80,
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

                # Speak when sentence complete + time it!
                if any(buffer.rstrip().endswith(p)
                       for p in ['.', '!', '?']):
                    tts_start = time.time()
                    speak(buffer.strip())
                    total_tts_time += time.time() - tts_start  # ← Track!
                    buffer = ""

                if chunk.get("done"):
                    break

        # Speak remaining!
        if buffer.strip():
            tts_start = time.time()
            speak(buffer.strip())
            total_tts_time += time.time() - tts_start  # ← Track!

        print()

        if full_response and len(full_response) > 5:
            response_cache[question_lower] = full_response
            print(f"💾 Cached!")

        return full_response, False, total_tts_time  # ← Return tts time!

    except Exception as e:
        print(f"❌ Error: {e}")
        return "Sorry, I had trouble thinking!", False, 0


def ask_llama(question):
    question_lower = question.lower().strip()

    # Fuzzy cache check!
    for cached_q, cached_ans in response_cache.items():
        ratio1 = fuzz.ratio(question_lower, cached_q)
        ratio2 = fuzz.partial_ratio(question_lower, cached_q)
        ratio3 = fuzz.token_sort_ratio(question_lower, cached_q)
        ratio4 = fuzz.token_set_ratio(question_lower, cached_q)
        best_score = max(ratio1, ratio2, ratio3, ratio4)

        if best_score >= CACHE_THRESHOLD:
            print(f"⚡ Cache hit! ({best_score}% match)")
            return cached_ans, True

    print(f"💬 Asking Llama...")
    try:
        prompt = f"Answer in 1-2 sentences only: {question}"
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "hf.co/puneetsiet2005/robotai-v2",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 80,
                    "temperature": 0.7,
                }
            },
            timeout=120
        )
        result = response.json().get("response", "").strip()

        if result and len(result) > 5:
            response_cache[question_lower] = result
            print(f"💾 Cached!")
            return result, False
        else:
            return "Sorry I could not find an answer!", False

    except Exception as e:
        print(f"❌ Error: {e}")
        return "Sorry, I had trouble thinking!", False

def speak(text):
    if not text or not text.strip():
        return
    try:
        piper_proc = subprocess.Popen(
            ['./piper/piper',
             '--model', 'en_US-ryan-low.onnx',
             '--output_raw',
             '--length_scale', '1.3'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        aplay_proc = subprocess.Popen(
            ['aplay', '-r', '22050', '-f', 'S16_LE',
             '-t', 'raw', '-D', 'plughw:0,0'],
            stdin=piper_proc.stdout,
            stderr=subprocess.DEVNULL
        )
        piper_proc.stdin.write(text.strip().encode())
        piper_proc.stdin.close()
        aplay_proc.wait()
    except Exception as e:
        print(f"TTS error: {e}")

def listen_for_wakeword():
    print("👂 Waiting for 'Hello'...")
    wake_words = [
        "hello", "hello robot", "hey robot",
        "hey", "hi", "hi robot"
    ]
    while True:
        audio = record_audio(duration=3)
        text = transcribe(audio).lower().strip()
        if text:
            print(f"   Heard: {text}")
            if any(word in text for word in wake_words):
                print("✅ Wake word detected!")
                return True

# Startup
print("Loading Whisper model...")
warmup_llama()

# Main loop
print("🤖 Robot Ready!")
speak("Hello! Say Hi to wake me up!")
question_count = 0

while True:
    listen_for_wakeword()
    speak("Yes!")

    for _ in range(3):
        question_count += 1

        # Check memory every 5 questions
        if question_count % 5 == 0:
            check_memory()

        # Timers
        total_start = time.time()

        audio = record_audio(duration=7)

        stt_start = time.time()
        text = transcribe(audio).strip()
        stt_time = time.time() - stt_start

        print(f"📝 You said: {text}")
        print(f"⏱️ STT time: {stt_time:.2f}s")

        if not text:
            speak("Please repeat!")
            continue

        if any(word in text.lower() for word in
               ["bye", "goodbye", "stop"]):
            speak("Goodbye!")
            break

        llama_start = time.time()
        response, from_cache, tts_time = ask_llama_streaming(text)
        llama_time = time.time() - llama_start

        tts_start = time.time()
        tts_time = time.time() - tts_start

        total_time = time.time() - total_start

        print(f"🤖 Robot: {response}")
        print(f"")
        print(f"⏱️ ====== TIMING BREAKDOWN ======")
        print(f"⏱️ STT (Whisper):  {stt_time:.2f}s")
        if from_cache:
            print(f"⏱️ LLM (Cache):    {llama_time:.2f}s ⚡")
        else:
            print(f"⏱️ LLM (Llama):    {llama_time:.2f}s")
        print(f"⏱️ TTS (Piper):    {tts_time:.2f}s")
        print(f"⏱️ Total response: {total_time:.2f}s")
        print(f"⏱️ ==============================")
