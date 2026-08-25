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
import urllib.error
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
        self.router = None

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

    # -- knowledge --------------------------------------------------

    def ask(self, question):
        """Answer a world-knowledge question using the existing shiva.py
        RAG + generation path.

        Deliberately does NOT call shiva.handle(): that router can dispatch
        to do_action() (which would drive motors outside this loop's budget)
        and it speaks internally, which would double up with SAY.
        """
        if self.dry_run and self.router is None:
            time.sleep(0.3)
            return f"(mock answer to: {question})"

        try:
            import shiva
        except Exception as e:
            return f"could not look that up ({e})"

        try:
            docs = shiva.retrieve(question)          # list, may be empty
        except Exception as e:
            return f"could not look that up ({e})"

        # GROUNDING: no retrieved context means no basis for an answer.
        # Generating anyway is how the robot invents facts.
        if not docs:
            return "I do not know that"

        context = "\n".join(docs)
        prompt = (
            "Answer the question in one short sentence using only the context.\n"
            "If the context does not contain the answer, say you do not know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        try:
            ans = shiva.generate(prompt).strip()
        except Exception as e:
            return f"could not look that up ({e})"

        return ans or "I do not know that"

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
Reply with EXACTLY ONE action line. No explanation. Never more than one line.

Actions:
LOOK: left | center | right   - point the camera and describe what is there
TURN: right                   - rotate the robot 90 degrees to face a new part of the room
ASK: <question>               - look up a fact you do not know
SAY: <text>                   - speak to the human
DONE:                         - stop

Rules:
- You may only report things that appeared in a "->" line above. Never guess.
- Use LOOK: for anything about the room. Use ASK: only for facts about the world.
- Look left, center and right. If the object is not there, TURN: right and look again.
- After you SAY something, the next line must be DONE:

Example:
Goal: find my red cup

LOOK: left
-> a wall and a lamp
LOOK: center
-> a closed door
LOOK: right
-> a red cup
SAY: I found your red cup
-> spoken
DONE:

Example:
Goal: turn left and tell me who invented the telephone

TURN: left
-> turned left, now facing a new part of the room
ASK: who invented the telephone
-> Alexander Graham Bell
SAY: Alexander Graham Bell invented the telephone
-> spoken
DONE:
"""


def build_prompt(goal, history):
    lines = [SYSTEM_TOOLS, "", f"Goal: {goal}", ""]
    for action, obs in history:
        if action != "(note)":
            lines.append(action)
        lines.append(f"-> {obs}")
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# MODEL CALL
# ----------------------------------------------------------------------

def ollama_generate(prompt, temperature=0.2):
    payload = json.dumps({
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 24,      # one line is all we want
            "stop": ["\n->", "\nGoal:"],
        },
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=STEP_TIMEOUT) as r:
            return json.loads(r.read())["response"]
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        raise RuntimeError(f"HTTP {e.code}: {body}") from None


# ----------------------------------------------------------------------
# PARSER
# ----------------------------------------------------------------------

VALID = {"LOOK", "TURN", "ASK", "SAY", "DONE"}
ACTION_RE = re.compile(r"^\s*(LOOK|TURN|ASK|SAY|DONE)\s*:\s*(.*)$", re.I)


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
        self.asks = 0
        self.turns_since_look = 0

    def allow_ask(self):
        if self.asks >= 3:
            return False, "refused: too many lookups, answer with what you have"
        self.asks += 1
        return True, None

    def allow_move(self, feet):
        if feet <= 0 or feet > 6:
            return False, f"refused: single move must be 1-6 feet"
        if self.distance + feet > MAX_DISTANCE:
            return False, f"refused: distance budget exhausted"
        self.distance += feet
        return True, None

    def allow_turn(self, deg):
        if self.turns_since_look >= 2:
            return False, "refused: you must LOOK: center before turning again"
        if self.turned + deg > MAX_TURN_DEG:
            return False, "refused: turn budget exhausted"
        self.turned += deg
        self.turns_since_look += 1
        return True, None


# ----------------------------------------------------------------------
# EXECUTOR
# ----------------------------------------------------------------------

def execute(hw, budget, action, arg):
    if action == "LOOK":
        d = arg.lower().strip().split()[0] if arg.strip() else ""
        if d not in ("left", "center", "right"):
            return "invalid direction, say LOOK: left or LOOK: center or LOOK: right"
        hw.point_neck(d)
        budget.turns_since_look = 0
        hw.speak("let me look")           # covers the vision latency
        return hw.see()

    if action == "ASK":
        q = arg.strip()
        if len(q) < 3:
            return "invalid, say ASK: followed by a question"
        ok, why = budget.allow_ask()
        if not ok:
            return why
        hw.speak("let me think")
        return hw.ask(q)

    if action == "TURN":
        m = re.search(r"(left|right)", arg.lower())
        direction = m.group(1) if m else "right"
        ok, why = budget.allow_turn(90)
        if not ok:
            return why
        hw.turn(direction, 90)
        return f"turned {direction}, now facing a new part of the room"

    return "unknown action"


# ----------------------------------------------------------------------
# THE LOOP
# ----------------------------------------------------------------------

def report(hw, history):
    """Terminal fallback: say what was actually observed, never invent."""
    seen = [obs for act, obs in history
            if act.startswith("LOOK") and not obs.startswith(("refused", "already", "invalid"))]
    if seen:
        hw.speak("I saw " + ", and ".join(seen[-3:]))
    else:
        hw.speak("I could not see anything useful.")


def react(goal, hw, verbose=True):
    history = []
    budget = Budget()
    seen_actions = []
    finished = False
    stuck_streak = 0

    for step in range(1, MAX_STEPS + 1):
        prompt = build_prompt(goal, history)

        # after a rejection, raise temperature to break the token groove
        temp = 0.2 if not stuck_streak else min(0.2 + 0.35 * stuck_streak, 0.9)
        try:
            raw = ollama_generate(prompt, temperature=temp)
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
            stuck_streak += 1
            continue

        line = f"{action}: {arg}".strip().rstrip(":")

        # loop breaker
        if line in seen_actions and action not in ("SAY",):
            repeats = seen_actions.count(line) + 1
            if action == "LOOK":
                tried = {a.split(":")[1].strip() for a in seen_actions
                         if a.startswith("LOOK")}
                untried = [d for d in ("left", "center", "right") if d not in tried]
                note = (f"that direction is done. Only the {untried[0]} side "
                        "is still unchecked."
                        if untried else
                        "all three directions here are done. Rotate to face a "
                        "new part of the room.")
            elif action == "TURN":
                note = "you have already rotated. Check what is in front of you."
            elif action == "ASK":
                note = "that lookup is already done. Report the answer above."
            else:
                note = "that step is already done. Finish now."

            if verbose:
                print(f"  -> [loop breaker x{repeats}] {note}")

            # CRITICAL: never write the rejected action line into history,
            # and never name the replacement in tool syntax. Either one makes
            # that token pattern dominant in context and the model copies it.
            # Use a neutral marker that build_prompt renders without a verb.
            if history and history[-1][0] == "(note)":
                history[-1] = ("(note)", note)
            else:
                history.append(("(note)", note))
            seen_actions.append(line)
            stuck_streak += 1

            # give up on steering and terminate with whatever was learned
            if repeats >= 3:
                if verbose:
                    print("[loop] stuck - terminating and reporting")
                report(hw, history)
                finished = True
                break
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
        stuck_streak = 0
        if verbose:
            print(f"  -> {obs}")
        history.append((line, obs))

        # a turn changes the heading, so previously-seen LOOK positions
        # are now genuinely new views. TURN entries must survive, or
        # repeated turns become invisible to the loop breaker.
        if action == "TURN" and not obs.startswith("refused"):
            seen_actions = [a for a in seen_actions if not a.startswith("LOOK")]

    if not finished:
        report(hw, history)

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