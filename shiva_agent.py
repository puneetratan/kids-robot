#!/usr/bin/env python3
"""
shiva_agent.py - ReAct loop for the Shiva robot.

The transcript IS the state. Each round we re-send the whole
action/observation history and ask the model for exactly ONE more line.

Run mocked (no hardware, no Ollama needed for the tools):
    python3 shiva_agent.py --dry-run --goal "find my water bottle"

Run for real on the Pi:
    python3 shiva_agent.py --goal "find my water bottle"
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL_NAME  = os.environ.get("AGENT_MODEL", "llama3.2:1b")

MAX_STEPS      = 12      # hard ceiling on loop iterations
MAX_DISTANCE   = 12.0   # total feet the agent may travel in one goal
MAX_TURN_DEG   = 720    # total degrees it may rotate in one goal
STEP_TIMEOUT   = 60     # seconds per model call

TRANSCRIPT_DIR = os.path.expanduser("~/kids-robot/agent_runs")


# ----------------------------------------------------------------------
# ADAPTER LAYER  <-- THE ONLY PART YOU NEED TO EDIT
#
# Point these at your real modules. Everything below this block is
# hardware-agnostic. If an import fails we fall back to mocks so the
# loop is testable on the Mac.
# ----------------------------------------------------------------------

DRY_RUN = False  # set by --dry-run


class Hardware:
    """Thin wrapper over your existing modules."""

    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.drive = None
        self.neck = None
        self.vision = None

        # mock world: robot starts at heading 0, bottle is at heading 180.
        # it will only be seen after the robot turns around.
        self.heading = 0
        self.neck_dir = "center"
        self._mock_world = {
            0:   {"left": "a white wall", "center": "a closed door", "right": "a bookshelf"},
            90:  {"left": "a bookshelf", "center": "a window with curtains", "right": "a floor lamp"},
            180: {"left": "a floor lamp", "center": "a sofa and a low table",
                  "right": "a blue water bottle on the floor next to a chair"},
            270: {"left": "a chair", "center": "a rug and some toys", "right": "a white wall"},
        }

        if dry_run:
            print("[hw] DRY RUN - all actuators mocked")
            return

        # ---- wire these to your real code -------------------------
        try:
            import drive as drive_mod            # your drive.py
            self.drive = drive_mod
        except Exception as e:
            print(f"[hw] drive.py unavailable ({e}) - mocking motion")

        try:
            import vision as vision_mod          # your vision.py
            self.vision = vision_mod
        except Exception as e:
            print(f"[hw] vision.py unavailable ({e}) - mocking vision")

        try:
            import servo_test as neck_mod        # or wherever pan/tilt lives
            self.neck = neck_mod
        except Exception as e:
            print(f"[hw] neck module unavailable ({e}) - mocking neck")
        # -----------------------------------------------------------

    # -- motion -----------------------------------------------------

    def forward(self, feet):
        """TODO: replace with your calibrated call, e.g.
        self.drive.forward(seconds=feet / FEET_PER_SEC)"""
        if self.drive is None:
            time.sleep(0.3)
            return f"moved forward about {feet} feet"
        self.drive.forward(feet)
        return f"moved forward about {feet} feet"

    def turn(self, direction, degrees):
        """TODO: replace with self.drive.turn_left(...) / turn_right(...)"""
        step = degrees if direction == "right" else -degrees
        self.heading = round((self.heading + step) / 90) * 90 % 360
        if self.drive is None:
            time.sleep(0.3)
            return f"turned {direction} {degrees} degrees"
        self.drive.turn(direction, degrees)
        return f"turned {direction} {degrees} degrees"

    # -- neck -------------------------------------------------------

    def point_neck(self, direction):
        """TODO: map to PCA9685 channel 0 (pan) angles.
        left=150, center=90, right=30 or whatever your clamp allows."""
        angles = {"left": 150, "center": 90, "right": 30}
        if direction not in angles:
            return f"cannot point neck {direction}"
        self.neck_dir = direction
        if self.neck is None:
            time.sleep(0.2)
            return f"neck facing {direction}"
        self.neck.set_pan(angles[direction])
        return f"neck facing {direction}"

    # -- vision -----------------------------------------------------

    def see(self):
        """TODO: replace with your qwen2.5vl call, e.g.
        return self.vision.describe(prompt='What objects are visible?')"""
        if self.vision is None:
            time.sleep(0.4)
            return self._mock_world[self.heading % 360][self.neck_dir]
        return self.vision.describe()

    # -- speech -----------------------------------------------------

    def speak(self, text):
        """TODO: route to your Piper TTS helper."""
        print(f"[speak] {text}")
        if self.dry_run:
            return
        try:
            import subprocess
            subprocess.run(
                f'echo {json.dumps(text)} | piper --model '
                f'/home/pi/piper/en_US-ryan-low.onnx --length_scale 1.3 '
                f'--output_raw | aplay -r 22050 -f S16_LE -t raw -',
                shell=True, check=False,
            )
        except Exception as e:
            print(f"[speak] failed: {e}")


# ----------------------------------------------------------------------
# PROMPT
# ----------------------------------------------------------------------

SYSTEM_TOOLS = """You control a small robot that can look around and turn in place.
Reply with EXACTLY ONE action line. No explanation.

Actions:
LOOK: left | center | right     - point the camera
SEE:                            - describe what the camera sees
TURN: right 90                  - rotate the whole robot
SAY: <text>                     - speak to the human
DONE:                           - the goal is complete

Rules:
- Only ONE line. Never more.
- LOOK: only points the camera. You must SEE: to observe anything.
- If you have looked left, center and right and not found it, TURN: right 90 and look again.
- When you find it, SAY: what you found, then DONE:
- If you have turned all the way around and not found it, SAY: you could not find it, then DONE:

Example:
Goal: find my red cup

LOOK: left
-> neck facing left
SEE:
-> a wall and a lamp
LOOK: right
-> neck facing right
SEE:
-> a red cup on a desk
SAY: I found your red cup on the desk
-> spoken
DONE:
"""


def build_prompt(goal, history):
    lines = [SYSTEM_TOOLS, "", f"Goal: {goal}", ""]
    for action, obs in history:
        lines.append(action)
        lines.append(f"-> {obs}")
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# MODEL CALL
# ----------------------------------------------------------------------

def ollama_generate(prompt):
    payload = json.dumps({
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 24,      # one line is all we want
            "stop": ["\n->", "\nGoal:"],
        },
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=STEP_TIMEOUT) as r:
        return json.loads(r.read())["response"]


# ----------------------------------------------------------------------
# PARSER
# ----------------------------------------------------------------------

VALID = {"LOOK", "SEE", "TURN", "MOVE", "SAY", "DONE"}
ACTION_RE = re.compile(r"^\s*(LOOK|SEE|TURN|MOVE|SAY|DONE)\s*:\s*(.*)$", re.I)


def parse(raw):
    """Return (ACTION, arg) or (None, reason). Takes the first valid line only."""
    if not raw:
        return None, "empty response"
    for line in raw.strip().splitlines():
        m = ACTION_RE.match(line)
        if m:
            return m.group(1).upper(), m.group(2).strip()
    return None, "no action line found"


# ----------------------------------------------------------------------
# BUDGET - this is the real mid-loop safety, not a content filter
# ----------------------------------------------------------------------

class Budget:
    def __init__(self):
        self.distance = 0.0
        self.turned = 0

    def allow_move(self, feet):
        if feet <= 0 or feet > 6:
            return False, f"refused: single move must be 1-6 feet"
        if self.distance + feet > MAX_DISTANCE:
            return False, f"refused: distance budget exhausted"
        self.distance += feet
        return True, None

    def allow_turn(self, deg):
        if self.turned + deg > MAX_TURN_DEG:
            return False, "refused: turn budget exhausted"
        self.turned += deg
        return True, None


# ----------------------------------------------------------------------
# EXECUTOR
# ----------------------------------------------------------------------

def execute(hw, budget, action, arg):
    if action == "LOOK":
        d = arg.lower().strip()
        if d not in ("left", "center", "right"):
            return "invalid direction, use left center or right"
        return hw.point_neck(d)

    if action == "SEE":
        hw.speak("let me look")           # covers the vision latency
        return hw.see()

    if action == "TURN":
        m = re.match(r"(left|right)\s*(\d+)?", arg.lower().strip())
        if not m:
            return "invalid turn, use: TURN: left 90"
        direction = m.group(1)
        degrees = int(m.group(2) or 90)
        ok, why = budget.allow_turn(degrees)
        if not ok:
            return why
        return hw.turn(direction, degrees)

    if action == "MOVE":
        m = re.search(r"[\d.]+", arg)
        if not m:
            return "invalid move, use: MOVE: 3"
        feet = float(m.group())
        ok, why = budget.allow_move(feet)
        if not ok:
            return why
        return hw.forward(feet)

    return "unknown action"


# ----------------------------------------------------------------------
# THE LOOP
# ----------------------------------------------------------------------

def react(goal, hw, verbose=True):
    history = []
    budget = Budget()
    seen_actions = []
    finished = False

    for step in range(1, MAX_STEPS + 1):
        prompt = build_prompt(goal, history)

        try:
            raw = ollama_generate(prompt)
        except Exception as e:
            print(f"[loop] model call failed: {e}")
            hw.speak("my brain is not responding")
            break

        action, arg = parse(raw)

        if verbose:
            print(f"\n--- step {step} ---")
            print(f"model: {raw.strip()!r}")

        # unparseable -> feed the error back, don't crash
        if action is None:
            note = "invalid, reply with ONE action line only"
            if verbose:
                print(f"  -> [rejected: {arg}] {note}")
            history.append((raw.strip()[:60] or "(empty)", note))
            continue

        line = f"{action}: {arg}".strip().rstrip(":")

        # loop breaker
        if line in seen_actions and action not in ("SEE", "SAY"):
            note = "already did that, try something different"
            if verbose:
                print(f"  -> [loop breaker] {note}")
            history.append((line, note))
            seen_actions.append(line)
            continue
        seen_actions.append(line)

        if action == "DONE":
            finished = True
            if verbose:
                print("[loop] agent signalled DONE")
            break

        if action == "SAY":
            hw.speak(arg)
            history.append((f"SAY: {arg}", "spoken"))
            continue

        obs = execute(hw, budget, action, arg)
        if verbose:
            print(f"  -> {obs}")
        history.append((line, obs))

        # a turn changes the heading, so previously-seen LOOK positions
        # are now genuinely new views
        if action == "TURN" and not obs.startswith("refused"):
            seen_actions = []

    if not finished:
        hw.speak("I could not finish that.")

    save_transcript(goal, history, budget, finished)
    return history


def save_transcript(goal, history, budget, finished):
    try:
        os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(TRANSCRIPT_DIR, f"{stamp}.json")
        with open(path, "w") as f:
            json.dump({
                "goal": goal,
                "finished": finished,
                "steps": len(history),
                "distance_ft": budget.distance,
                "turned_deg": budget.turned,
                "history": history,
            }, f, indent=2)
        print(f"\n[log] transcript -> {path}")
    except Exception as e:
        print(f"[log] could not save: {e}")


# ----------------------------------------------------------------------

def main():
    global DRY_RUN
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", default="find my water bottle")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", default=MODEL_NAME)
    args = ap.parse_args()

    DRY_RUN = args.dry_run
    globals()["MODEL_NAME"] = args.model

    print(f"model: {MODEL_NAME}   goal: {args.goal!r}")
    hw = Hardware(dry_run=args.dry_run)

    t0 = time.time()
    history = react(args.goal, hw)
    print(f"\n[done] {len(history)} rounds in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()