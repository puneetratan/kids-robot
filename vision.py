"""
vision.py
===========
Camera capture + vision-language description, with the neck
moving while the model thinks.

WHY THE NECK MOVES
------------------
Inference takes 27-32s on a Pi 5. Silence for that long reads
as a crash. A robot that visibly looks at the thing it is
examining buys the whole wait - kids read it as thinking rather
than broken. The movement is theatre, but honest theatre: it
really is looking.

MODEL CHOICE
------------
qwen2.5vl:7b, chosen over llava:7b and moondream on accuracy.

Same dinosaur, same image, same machine:
    moondream   "urn" - a broken single-word output on
                question-shaped prompts, and invented context
                when it did answer
    llava:7b    stegosaurus, velociraptor, Allosaurus - all
                wrong, and different every run
    qwen2.5vl   sauropod / brachiosaurus / diplodocus /
                Apatosaurus - correct family every run

Qwen is no faster than llava on the Pi but it is consistently
right, and consistency is what matters when a child is asking.

Worth keeping in mind: a GPU made this 20-40x faster and no
more accurate - the Mac invented species too. Hardware fixes
latency; only a better model fixes hallucination.

IMAGE SIZE
----------
384px. 512 produced a 188s outlier on the Pi against 27s at
384. The variance is worse than the average, and unpredictable
latency is harder to design around than slow latency.

DIRECTION
---------
"look left" pans the head and then describes what is there.
Direction is a parameter of looking, not a separate class -
the wheels turn on "turn left", the head turns on "look left".

The head stays where it was aimed rather than recentring, so a
follow-up question can reuse the view.

REMOTE INFERENCE
----------------
    VISION_HOST=http://192.168.1.140:11434 python3 shiva.py

runs vision on a laptop and leaves everything else on the Pi.
Unset, it all runs locally.
"""

import base64
import os
import random
import subprocess
import threading
import time

import requests
from PIL import Image

OLLAMA_HOST = os.getenv("VISION_HOST",
                        os.getenv("OLLAMA_HOST", "http://localhost:11434"))
MODEL = os.getenv("VISION_MODEL", "qwen2.5vl:7b")

CAPTURE_PATH = "/tmp/vision.jpg"
RESIZED_PATH = "/tmp/vision_small.jpg"
IMAGE_SIZE = 384

PAN, TILT = 0, 1
CENTER_PAN, CENTER_TILT = 90, 90

LOOKING_PHRASES = [
    "Let me take a look at that.",
    "Ooh, let me see!",
    "Hold it still, I'm looking.",
    "Let me have a good look.",
]

# Descriptive framing, plus an explicit instruction not to guess.
# On the Mac, the two prompts carrying that instruction stayed
# accurate while the bare "what is this object?" prompts invented
# a species. Cheapest grounding intervention available - the same
# principle as the RAG prompt, applied to pixels.
PROMPT_TEMPLATE = (
    "Describe what you can clearly see in this image. "
    "Only mention things that are actually visible. "
    "Do not guess or add details you cannot see. "
    "Answer in two or three short sentences for a young child.\n\n"
    "The child asked: {question}"
)

CANT_SEE = "Hmm, I can't see that clearly. Can you hold it closer?"


def _kill_camera_holders():
    """Only one process can hold the camera. A leftover
    rpicam-vid stream will block capture."""
    subprocess.run(["pkill", "rpicam"], stderr=subprocess.DEVNULL)
    time.sleep(0.5)


def capture(path=CAPTURE_PATH):
    """Autofocus needs time to settle - a short capture gives
    blurred frames indoors. --awb indoor fixes a blue cast."""
    _kill_camera_holders()
    subprocess.run(
        ["rpicam-jpeg", "-o", path, "-n",
         "--awb", "indoor",
         "--autofocus-mode", "auto",
         "-t", "3000"],
        check=True, stderr=subprocess.DEVNULL,
    )
    return path


def _resize(src=CAPTURE_PATH, dst=RESIZED_PATH, size=IMAGE_SIZE):
    im = Image.open(src)
    im.thumbnail((size, size))
    im.save(dst)
    return dst


def _describe(question, image_path):
    img = base64.b64encode(open(image_path, "rb").read()).decode()
    r = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": MODEL,
            "prompt": PROMPT_TEMPLATE.format(question=question),
            "images": [img],
            "stream": False,
        },
        timeout=300,
    )
    return r.json().get("response", "").strip()


# ============================================================
# Direction
#
# "look left" routes VISION, not ACTION - direction is a
# parameter of looking rather than a separate class. The wheels
# turn on "turn left"; the head turns on "look left".
#
# Pan is mirrored: from the robot's point of view its own left
# is a higher servo angle. A child saying "look left" means the
# robot's left, which is what the classifier examples assume.
# ============================================================
PAN_LEFT, PAN_RIGHT = 140, 40
TILT_UP, TILT_DOWN = 60, 125

DIRECTIONS = [
    (("left",),  PAN_LEFT,   None),
    (("right",), PAN_RIGHT,  None),
    (("up", "ceiling", "above", "sky"), None, TILT_UP),
    (("down", "floor", "below", "ground"), None, TILT_DOWN),
]


def parse_direction(question):
    """Returns (pan, tilt), either may be None for 'unchanged'.
    First match wins - a question mentioning both is rare enough
    not to be worth handling."""
    lowered = question.lower()
    for keywords, pan, tilt in DIRECTIONS:
        if any(k in lowered for k in keywords):
            return pan, tilt
    return None, None


def aim(kit, pan=None, tilt=None, settle=1.2):
    """Point the head, then let it settle. Capturing mid-travel
    gives a motion-blurred frame, which is exactly the input
    that makes a VLM invent things."""
    if kit is None:
        return
    try:
        if pan is not None:
            kit.servo[PAN].angle = max(0, min(180, pan))
        if tilt is not None:
            kit.servo[TILT].angle = max(0, min(180, tilt))
        if pan is not None or tilt is not None:
            time.sleep(settle)
    except Exception:
        pass


class NeckIdle:
    """Small head movements while the model runs. Runs in a
    thread so it does not add to the wait - it fills it."""

    def __init__(self, kit, pan=CENTER_PAN, tilt=CENTER_TILT):
        self.kit = kit
        self.pan = pan
        self.tilt = tilt
        self._stop = threading.Event()
        self._thread = None

    def _wander(self):
        # Small offsets only. Large sweeps look like the robot
        # has lost the object rather than studying it.
        offsets = [(0, 0), (-12, 4), (8, -6), (-6, -4), (10, 5), (0, 0)]
        i = 0
        while not self._stop.is_set():
            dp, dt = offsets[i % len(offsets)]
            try:
                self.kit.servo[PAN].angle = max(0, min(180, self.pan + dp))
                self.kit.servo[TILT].angle = max(0, min(180, self.tilt + dt))
            except Exception:
                return
            i += 1
            self._stop.wait(1.6)

    def __enter__(self):
        if self.kit is not None:
            self._thread = threading.Thread(target=self._wander, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        # Stay where it was told to look. Snapping back to
        # centre would undo the aim before a follow-up question
        # like "what colour is it?" could reuse it.
        if self.kit is not None:
            try:
                self.kit.servo[PAN].angle = self.pan
                self.kit.servo[TILT].angle = self.tilt
            except Exception:
                pass


def look_and_describe(question, speak=None, kit=None):
    """Aim, capture, describe, return the answer.

    speak: callable used for the holding phrase. Passing
           shiva's speak() means the robot talks before the
           long wait rather than going silent.
    kit:   ServoKit instance. Omit and the head stays still.
    """
    pan, tilt = parse_direction(question)

    if speak:
        speak(random.choice(LOOKING_PHRASES))

    # Aim before capturing. Doing it after would describe
    # wherever the head happened to be pointing, which is what
    # made "look left" and "look right" return the same answer.
    aim(kit, pan, tilt)

    try:
        capture()
    except subprocess.CalledProcessError:
        return CANT_SEE

    path = _resize()

    with NeckIdle(kit,
                  pan if pan is not None else CENTER_PAN,
                  tilt if tilt is not None else CENTER_TILT):
        try:
            answer = _describe(question, path)
        except requests.Timeout:
            return "That took too long to look at. Try again?"
        except Exception:
            return CANT_SEE

    return answer or CANT_SEE


if __name__ == "__main__":
    try:
        from adafruit_servokit import ServoKit
        kit = ServoKit(channels=16)
    except Exception as e:
        print(f"no servos ({e}) - head will stay still")
        kit = None

    print(f"model: {MODEL} @ {OLLAMA_HOST}")
    t = time.time()
    print(look_and_describe("What is this?", speak=print, kit=kit))
    print(f"\n{time.time() - t:.1f}s total")
