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

# The agent LLM can live on a different machine from the rest of the robot.
# Keep it separate from OLLAMA_HOST: shiva.py uses that one for ASK/RAG with
# its own fine-tuned model, which does not exist on the remote host.
AGENT_HOST  = os.environ.get("AGENT_HOST", "http://127.0.0.1:11434")
MODEL_NAME  = os.environ.get("AGENT_MODEL", "llama3.2:1b")

# used if AGENT_HOST is unreachable - degraded, not dead
FALLBACK_HOST  = "http://127.0.0.1:11434"
FALLBACK_MODEL = os.environ.get("AGENT_FALLBACK_MODEL", "llama3.2:1b")

MAX_STEPS      = 20   # 12 views + 4 turns + report      # hard ceiling on loop iterations
MAX_DISTANCE     = 6.0   # total feet per goal - small room, no sensors
MAX_SINGLE_MOVE  = 2.0   # never cross a room in one blind burst
BACK_FEET        = 1.0   # fixed retreat distance
MAX_BACKUPS      = 2     # no rear sensor, so reversing is strictly limited
STOP_MARGIN      = 1.0   # feet of clearance the robot refuses to give up
MAX_TURN_DEG   = 720    # total degrees it may rotate in one goal
STEP_TIMEOUT   = 60
TIMING         = os.environ.get("AGENT_TIMING", "1") != "0"     # seconds per model call

TRANSCRIPT_DIR = os.path.expanduser("~/kids-robot/agent_runs")
FRAME_DIR      = os.path.expanduser("~/kids-robot/agent_runs/frames")
SAVE_FRAMES    = os.environ.get("SAVE_FRAMES", "1") != "0"
RUN_ID         = datetime.now().strftime("%Y%m%d-%H%M%S")


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

    def __init__(self, dry_run=False, mute=False):
        self.dry_run = dry_run
        self.mute = mute
        self.drive = None
        self.neck = None
        self.vision = None
        self.router = None
        self._tts = None
        self._kit = None
        self._sonar = None
        self._warned_voice = False
        self._frame_n = 0
        self._mock_distance = 4.0

        # mock world: robot starts at heading 0, bottle is at heading 180.
        # it will only be seen after the robot turns around.
        self.heading = 0
        self.position = 0        # bumps on every MOVE/BACK
        self.neck_dir = "center"
        self._mock_world = {
            0:   {"left": "a white wall", "center": "a closed door",
                  "right": "a bookshelf", "down": "bare carpet",
                  "up": "a ceiling light"},
            90:  {"left": "a bookshelf", "center": "a window with curtains",
                  "right": "a floor lamp", "down": "a rug", "up": "a ceiling"},
            180: {"left": "a floor lamp", "center": "a sofa and a low table",
                  "right": "a blue water bottle next to a chair",
                  "down": "a yellow toy car on the carpet", "up": "a ceiling"},
            270: {"left": "a chair", "center": "a rug and some toys",
                  "right": "a white wall", "down": "some toys on the floor",
                  "up": "a ceiling"},
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

    # Calibrate once: run scripts/calibrate_turn.py, mark the floor, adjust.
    # Accuracy is not critical - the loop counts areas A-D by counting turns,
    # not by measuring them. It only matters if MOVE is added later.
    SECONDS_PER_90 = float(os.environ.get("TURN_SECONDS_90", "0.6"))
    FEET_PER_SEC   = float(os.environ.get("DRIVE_FEET_PER_SEC", "1.5"))

    def forward(self, feet):
        """Drive forward in short slices, checking the sonar between each.

        A single timed burst is blind: anything that appears after the burst
        starts gets hit. Slicing the move and re-reading distance lets the
        robot stop mid-move, which is the whole point of having a sensor.
        Returns (feet_actually_travelled, stopped_early).
        """
        if self.dry_run:
            time.sleep(0.3)
            return feet, False
        try:
            import drive
        except Exception as e:
            print(f"[hw] cannot move ({e})")
            return 0.0, False

        total = min(feet / self.FEET_PER_SEC, 6.0)   # seconds
        slice_s = 0.15
        elapsed = 0.0
        stopped = False
        try:
            drive.forward()
            while elapsed < total:
                time.sleep(slice_s)
                elapsed += slice_s
                d = self.distance()
                if d is not None and d < STOP_MARGIN:
                    stopped = True
                    break
        finally:
            drive.stop()
        return round(elapsed * self.FEET_PER_SEC, 1), stopped

    def backward(self, feet):
        if self.dry_run:
            time.sleep(0.3)
            return f"backed up {feet} feet"
        try:
            import drive
        except Exception as e:
            return f"cannot move ({e})"
        try:
            drive.backward()
            time.sleep(min(feet / self.FEET_PER_SEC, 3.0))
        finally:
            drive.stop()
        return f"backed up {feet} feet"

    def turn(self, direction, degrees):
        step = degrees if direction == "right" else -degrees
        self.heading = round((self.heading + step) / 90) * 90 % 360
        if self.dry_run:
            time.sleep(0.3)
            return f"turned {direction} {degrees} degrees"
        try:
            import drive
        except Exception as e:
            return f"cannot turn ({e})"
        try:
            (drive.turn_right if direction == "right" else drive.turn_left)()
            time.sleep(min(degrees / 90.0 * self.SECONDS_PER_90, 4.0))
        finally:
            drive.stop()                 # stop even if interrupted
        time.sleep(0.3)                  # let the chassis settle before looking
        return f"turned {direction} {degrees} degrees"

    # -- neck -------------------------------------------------------

    def _servo_kit(self):
        """Get a ServoKit without depending on servo_test.py (a demo script).
        Prefer one that already exists so we do not open I2C twice."""
        if self._kit is not None:
            return self._kit
        try:
            import vision
            self._kit = getattr(vision, "kit", None)
        except Exception:
            pass
        if self._kit is None:
            try:
                import servo_test
                self._kit = getattr(servo_test, "kit", None)
            except Exception:
                pass
        if self._kit is None:
            from adafruit_servokit import ServoKit
            self._kit = ServoKit(channels=16)
        return self._kit

    def point_neck(self, direction):
        """Aim the head using vision.py's own direction table.

        vision.py already resolves the mirror problem (a child's "left" is a
        HIGHER pan angle), so reuse it rather than duplicating angles here.
        """
        self.neck_dir = direction
        if self.dry_run:
            time.sleep(0.2)
            return f"neck facing {direction}"

        try:
            import vision
        except Exception as e:
            return f"cannot move neck ({e})"

        # Tilt depends on how the camera is physically mounted, so these are
        # overridable. On this build a HIGHER angle points LOWER: vision.py's
        # default of 125 for "down" still showed a standing person, meaning
        # the whole range sat above the floor. Calibrate before trusting it.
        SIDE_PAN = {"left": 140, "center": 90, "right": 40}
        HEIGHT_TILT = {
            "up":    float(os.environ.get("TILT_UP", "60")),
            "level": float(os.environ.get("TILT_LEVEL", "90")),
            "down":  float(os.environ.get("TILT_DOWN", "125")),
        }

        words = direction.split()
        side = next((w for w in words if w in SIDE_PAN), None)
        height = next((w for w in words if w in HEIGHT_TILT), None)

        # A side with no height keeps the current tilt; a height with no side
        # keeps the current pan. Naming both aims at one of nine cells.
        pan = SIDE_PAN.get(side)
        tilt = HEIGHT_TILT.get(height)

        # prefer vision.py's own table for the sides so the agent and the
        # voice pipeline never disagree about which way "left" is
        if side:
            for keywords, p, _t in getattr(vision, "DIRECTIONS", []):
                if side in keywords and p is not None:
                    pan = p
                    break

        # up/down set tilt only and leave pan None, which is valid -
        # vision.aim() treats None as "leave that axis alone".
        if pan is None and tilt is None:
            return f"cannot point neck {direction}"

        try:
            vision.aim(self._servo_kit(), pan=pan, tilt=tilt)
        except Exception as e:
            return f"neck movement failed ({e})"
        return f"neck facing {direction}"

    # -- vision -----------------------------------------------------

    def see(self):
        """Describe what the camera sees RIGHT NOW.

        Deliberately does not call vision.look_and_describe(): that parses a
        direction out of the question and re-aims the head, which would fight
        with point_neck() having already aimed. Capture where we are pointed.
        """
        if self.dry_run:
            time.sleep(0.4)
            w = self._mock_world[self.heading % 360]
            parts = self.neck_dir.split()
            key = next((p for p in parts if p in ("up", "down")), None) \
                or next((p for p in parts if p in w), "center")
            return w.get(key, "nothing in particular")

        try:
            import vision
        except Exception as e:
            return f"cannot see ({e})"

        question = "What objects are in this image? List them briefly."
        try:
            _t = time.time()
            vision.capture()
            _t_cap = time.time()
            vision._resize()
            _t_res = time.time()
            desc = vision._describe(question, vision.RESIZED_PATH)
            if TIMING:
                print(f"       [t]   capture {_t_cap-_t:.1f}s  "
                      f"resize {_t_res-_t_cap:.1f}s  "
                      f"model {time.time()-_t_res:.1f}s")

            # Keep every frame. vision.py overwrites one file, so without
            # this you can only ever inspect the last thing the robot saw -
            # which is useless when trying to work out why it missed
            # something six steps ago.
            if SAVE_FRAMES:
                try:
                    import shutil
                    os.makedirs(FRAME_DIR, exist_ok=True)
                    self._frame_n += 1
                    tag = self.neck_dir.replace(" ", "-")
                    dst = os.path.join(
                        FRAME_DIR,
                        f"{RUN_ID}_{self._frame_n:02d}_h{self.heading}_{tag}.jpg")
                    shutil.copy(vision.RESIZED_PATH, dst)
                except Exception:
                    pass
        except Exception as e:
            # fall back to the public helper if the private path changed
            try:
                desc = vision.look_and_describe(question, speak=None,
                                                kit=self._servo_kit())
            except Exception as e2:
                return f"cannot see ({e}; {e2})"

        desc = (desc or "").strip().replace("\n", " ")

        # vision.py returns a kid-facing fallback ("hold it closer") when it
        # cannot describe a frame. That message is for a child holding an
        # object up to the camera, not for a robot searching a room - and if
        # it reaches the transcript the agent treats it as a description, and
        # can even read it back as the final answer. Make it an honest miss.
        low = desc.lower()
        if (not desc
                or "hold it closer" in low
                or "see that clearly" in low):
            return "nothing identifiable from here"

        return desc[:300]

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
        """Speak via piper directly.

        Deliberately does NOT `import shiva` - that loads Whisper and the
        classifier (20s+ on the Pi) just to say four words, and it stalls
        the loop mid-goal.

        APLAY_DEVICE=pipewire routes to a Bluetooth speaker through
        PipeWire. Leave it unset for a directly attached USB speaker.
        """
        print(f"[speak] {text}")
        if self.dry_run or self.mute:
            return

        piper = os.environ.get("PIPER_BIN",
                               os.path.expanduser("~/kids-robot/piper/piper"))
        voice = os.environ.get(
            "PIPER_VOICE",
            os.path.expanduser("~/kids-robot/piper/en_US-ryan-low.onnx"))
        if not os.path.exists(voice):
            # Silent failure here cost real debugging time: the robot just
            # goes mute with no clue why. Say it once, out loud, in the log.
            if not self._warned_voice:
                print(f"[speak] voice model not found at {voice} - "
                      "set PIPER_VOICE. Continuing without audio.")
                self._warned_voice = True
            return

        import subprocess
        try:
            p1 = subprocess.run(
                [piper, "--model", voice, "--length_scale", "1.3",
                 "--output_raw"],
                input=text.encode(), stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, timeout=20)

            dev = os.environ.get("APLAY_DEVICE")
            cmd = ["aplay", "-q"]
            if dev:
                cmd += ["-D", dev]
            cmd += ["-r", "22050", "-f", "S16_LE", "-c", "1", "-t", "raw", "-"]
            subprocess.run(cmd, input=p1.stdout,
                           stderr=subprocess.DEVNULL, timeout=20)
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[speak] failed: {e}")


# ----------------------------------------------------------------------
# PROMPT
# ----------------------------------------------------------------------

SYSTEM_TOOLS = """You control a small robot that can look around and turn in place.
Reply with EXACTLY ONE action line. No explanation. Never more than one line.

Actions:
LOOK: <side> <height>         - point the camera and describe what is there\n                                (side: left, center, right)\n                                (height: up, level, down - optional)\n                                e.g. LOOK: left down
TURN: <direction>             - rotate the robot 90 degrees\n                                (direction is one of: left, right)
MOVE: <feet>                  - drive forward (only right after looking ahead)
BACK: <feet>                  - reverse out of a tight spot
ASK: <question>               - look up a fact about the world
SAY: <text>                   - speak the answer to the human
DONE:                         - stop

The room has four areas: A, B, C, D. One TURN moves you to the next area.
Two turns in the same direction put you in the area behind you.

Rules:
- You may only report things that appeared in a "->" line above. Never guess.
- The robot is blind while driving. Before MOVE you must LOOK: center, and
  you may only move once per look. Move 1 or 2 feet at a time.
- If something is in the way, BACK: 1 and turn.
- As soon as a "->" line answers the goal, SAY: it. Do not keep looking.
- Only keep searching if the goal names a specific object you have not found.
- After SAY: the next line must be DONE:

Example - facing behind you:
Goal: look behind you and tell me what you see

TURN: right
-> turned right, now in area B. Areas not yet searched: A, B, C, D
TURN: right
-> turned right, now in area C. Areas not yet searched: A, B, C, D
LOOK: center
-> a sofa and a low table
SAY: behind me I see a sofa and a low table
-> spoken
DONE:

Example - reporting what is there:
Goal: turn left and tell me what you see

TURN: left
-> turned left, now facing a new part of the room
LOOK: center
-> a sofa and a low table
SAY: I see a sofa and a low table
-> spoken
DONE:

Example - approaching something:
Goal: go to the red chair

LOOK: center
-> a red chair a few feet ahead
MOVE: 2
-> moved forward 2.0 feet. You cannot see what is ahead until you observe again
LOOK: center
-> the red chair is right in front of the robot
SAY: I am at the red chair
-> spoken
DONE:

Example - searching for something:
Goal: find my red cup

LOOK: center down
-> bare floor
LOOK: center level
-> a wall and a lamp
LOOK: center up
-> a ceiling
TURN: left
-> turned left, now in area D
LOOK: center down
-> a red cup on the floor
SAY: I found your red cup
-> spoken
DONE:

Example - looking up a fact (notice ASK repeats the goal's question):
Goal: turn right and tell me about <subject>

TURN: right
-> turned right, now in area B
ASK: tell me about <subject>
-> <the answer comes back here>
SAY: <report the answer above>
-> spoken
DONE:
"""


def build_prompt(goal, history):
    lines = [SYSTEM_TOOLS, "", f"Goal: {goal}", ""]
    for action, obs in history:
        if action not in ("(note)", "(hint)"):
            lines.append(action)
        lines.append(f"-> {obs}")
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# MODEL CALL
# ----------------------------------------------------------------------

def _post(host, model, prompt, temperature):
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 48,
            "stop": ["\n->", "\nGoal:"],
        },
    }).encode()
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=STEP_TIMEOUT) as r:
            return json.loads(r.read())["response"]
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        raise RuntimeError(f"HTTP {e.code} from {host}: {body}") from None


_fell_back = {"done": False}


def ollama_generate(prompt, temperature=0.2):
    try:
        return _post(AGENT_HOST, MODEL_NAME, prompt, temperature)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        if AGENT_HOST == FALLBACK_HOST:
            raise
        if not _fell_back["done"]:
            print(f"[llm] {AGENT_HOST} unreachable ({e}) - "
                  f"falling back to local {FALLBACK_MODEL}")
            _fell_back["done"] = True
        return _post(FALLBACK_HOST, FALLBACK_MODEL, prompt, temperature)


# ----------------------------------------------------------------------
# PARSER
# ----------------------------------------------------------------------

VALID = {"LOOK", "TURN", "MOVE", "BACK", "ASK", "SAY", "DONE"}
ACTION_RE = re.compile(r"^\s*(LOOK|TURN|MOVE|BACK|ASK|SAY|DONE)\s*:\s*(.*)$", re.I)


def parse_all(raw):
    """Return every valid action line, in order.

    A capable model often emits a short plan in one response
    ("TURN: left / TURN: left / LOOK: center"). Discarding all but the
    first line throws away correct work and forces the model to
    re-derive the plan every round.
    """
    if not raw:
        return []
    out = []
    for line in raw.strip().splitlines():
        m = ACTION_RE.match(line)
        if m:
            out.append((m.group(1).upper(), m.group(2).strip()))
    return out


def parse(raw):
    """First action only. Kept for callers that want a single step."""
    acts = parse_all(raw)
    return acts[0] if acts else (None, "no action line found")


# ----------------------------------------------------------------------
# BUDGET - this is the real mid-loop safety, not a content filter
# ----------------------------------------------------------------------

class Budget:
    def __init__(self):
        self.distance = 0.0
        self.turned = 0
        self.asks = 0
        self.backed = 0
        self.turns_since_look = 0
        self.moved = 0.0
        self.looked_forward = False   # set by LOOK: center, cleared by MOVE

    def allow_move(self, feet):
        """No obstacle sensors: the ONLY evidence the path is clear is a
        camera frame taken just now, facing forward. So a move requires a
        fresh LOOK: center, and one move is allowed per look."""
        if not self.looked_forward:
            return False, ("refused: nothing is known about the space ahead. "
                           "Observe straight ahead first.")
        if feet <= 0 or feet > MAX_SINGLE_MOVE:
            return False, f"refused: one move may be at most {MAX_SINGLE_MOVE} feet"
        if self.moved + feet > MAX_DISTANCE:
            return False, "refused: no distance left in this goal"
        self.moved += feet
        self.looked_forward = False    # must look again before moving again
        return True, None

    def allow_back(self):
        if self.backed >= MAX_BACKUPS:
            return False, ("refused: no more reversing. There is no rear "
                           "sensor. Turn and observe instead.")
        self.backed += 1
        return True, None

    def allow_ask(self):
        if self.asks >= 3:
            return False, "refused: no more lookups. Report what you already know."
        self.asks += 1
        return True, None

    def allow_turn(self, deg, planned=False):
        # A run of turns emitted as ONE plan is deliberate reasoning
        # ("four turns = all the way around"). A run of turns emitted as
        # separate reactive calls is fixation. Only cap the second kind.
        if self.turns_since_look >= 2 and not planned:
            return False, "refused: rotating again is not allowed until you observe what is in front of you"
        if self.turned + deg > MAX_TURN_DEG:
            return False, "refused: turn budget exhausted"
        self.turned += deg
        self.turns_since_look += 1
        return True, None


# ----------------------------------------------------------------------
# EXECUTOR
# ----------------------------------------------------------------------

AREA = {0: "A", 90: "B", 180: "C", 270: "D"}


def execute(hw, budget, action, arg, searched=None, planned=False):
    if action == "LOOK":
        d = arg.lower().strip()
        # Reject menu echoes like "left | center | right".
        if "|" in d or "," in d:
            return "invalid: choose one direction, not a list"
        words = d.split()
        SIDES = ("left", "center", "right")
        HEIGHTS = ("up", "down", "level")
        if len(words) > 2:
            return "invalid: use a side and optionally a height, e.g. left down"
        side = next((w for w in words if w in SIDES), None)
        height = next((w for w in words if w in HEIGHTS), None)
        if side is None and height is None:
            return ("invalid: say a side (left, center, right) and optionally "
                    "a height (up, down)")
        d = " ".join(x for x in (side, height) if x)
        t0 = time.time()
        hw.point_neck(d)
        # Observing clears the consecutive-turn cap. Without this the counter
        # only ever climbs and the robot can never rotate again after two
        # turns - it silently loses the ability to finish a sweep.
        budget.turns_since_look = 0
        t1 = time.time()
        hw.speak("let me look")           # covers the vision latency
        t2 = time.time()
        obs = hw.see()
        if d == "center" and not obs.startswith(("cannot", "invalid")):
            budget.looked_forward = True
        t3 = time.time()
        if TIMING:
            print(f"       [t] neck {t1-t0:.1f}s  say {t2-t1:.1f}s  "
                  f"vision {t3-t2:.1f}s")
        return obs

    if action == "MOVE":
        m = re.fullmatch(r"[\d.]+", arg.strip())
        if not m:
            return "invalid: say MOVE followed by a number of feet, e.g. MOVE: 2"
        feet = float(m.group())

        clear = hw.distance()
        if clear is not None:
            budget.looked_forward = True        # measurement beats a photo
            if clear < STOP_MARGIN:
                return (f"refused: something is {clear:.1f} feet ahead. "
                        "Back up or turn.")
            feet = min(feet, clear - STOP_MARGIN)
            if feet < 0.3:
                return (f"refused: only {clear:.1f} feet of clearance ahead")

        ok, why = budget.allow_move(feet)
        if not ok:
            return why
        travelled, stopped = hw.forward(feet)
        hw.position += 1
        after = hw.distance()
        if stopped:
            return (f"stopped after {travelled} feet - something appeared "
                    "ahead while moving. Turn or back up.")
        tail = (f" {after:.1f} feet of clear space ahead now."
                if after is not None else
                " You cannot see what is ahead until you observe again.")
        return f"moved forward {travelled} feet.{tail}"

    if action == "BACK":
        # Escape hatch: reversing is permitted without a look because the
        # robot just came from there. But there is NO rear sensor, so it is
        # capped hard and cannot be repeated indefinitely.
        if arg.strip() and not re.fullmatch(r"[\d.]*", arg.strip()):
            return "invalid: say BACK on its own, with no extra words"
        ok, why = budget.allow_back()
        if not ok:
            return why
        hw.backward(BACK_FEET)
        hw.position += 1
        budget.looked_forward = False
        return (f"backed up {BACK_FEET} feet. There is no rear sensor, "
                "so observe before moving again")

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
        d = arg.lower().strip()
        if "|" in d or len(d.split()) > 1:
            return "invalid: choose exactly one direction, not a list"
        m = re.fullmatch(r"(left|right)", d)
        if not m:
            return "invalid: the direction must be left or right"
        direction = m.group(1)
        ok, why = budget.allow_turn(90, planned=planned)
        if not ok:
            return why
        hw.turn(direction, 90)
        here = AREA[hw.heading % 360]
        # NOT "part 4 of 4" - an ordinal like that reads as a progress
        # counter and the model concludes the search is finished. Name the
        # area, then state explicitly what is still unsearched.
        remaining = [v for k, v in sorted(AREA.items())
                     if v not in (searched or set())]
        if remaining:
            return (f"turned {direction}, now in area {here}. "
                    f"Areas not yet searched: {', '.join(remaining)}")
        return f"turned {direction}, now in area {here}. Every area has been searched"

    return "unknown action"


# ----------------------------------------------------------------------
# THE LOOP
# ----------------------------------------------------------------------

SEARCH_RE = re.compile(
    r"\b(find|search for|look for|locate|where is|where's|is there)\b", re.I)
STOPWORDS = {"my", "the", "a", "an", "in", "this", "that", "room", "here",
             "please", "for", "is", "any", "some", "your", "you", "me"}


def goal_target(goal):
    """If the goal is a search, return the words that identify the target.

    A report goal ("tell me what you see") is satisfied by ANY observation.
    A search goal is satisfied only by an observation containing the target,
    so the two need different stopping rules.
    """
    if not SEARCH_RE.search(goal):
        return None
    tail = SEARCH_RE.split(goal, maxsplit=1)[-1]
    words = {w.strip(".,?!").lower() for w in tail.split()}
    words -= STOPWORDS
    return {w for w in words if len(w) > 2} or None


def target_found(target, obs):
    """True only when the observation matches the target strongly enough.

    A single word is far too weak. "a pink object that looks like a toy"
    matches "toy" while being neither yellow nor a car, and one loose hit is
    enough to make the robot announce success. So: require at least TWO of
    the target words, and if a colour was asked for it must be one of them.

    Word-boundary matching with a short suffix allowance, so "car" matches
    "cars" but not "carpet".
    """
    if not target:
        return False

    COLOURS = {"red", "blue", "green", "yellow", "pink", "purple", "orange",
               "black", "white", "brown", "grey", "gray", "silver", "gold"}

    def present(t):
        return bool(re.search(rf"\b{re.escape(t)}(s|es|ing)?\b", obs, re.I))

    hits = {t for t in target if present(t)}
    if len(hits) < min(2, len(target)):
        return False

    # If the goal named a colour, that colour must actually be in the frame.
    wanted_colours = target & COLOURS
    if wanted_colours and not (wanted_colours & hits):
        return False
    return True


def ungrounded_claims(text, goal, history):
    """Named entities in a SAY that came from nowhere the robot looked.

    The model will happily answer world questions from its own weights -
    which are frozen, often stale, and unverifiable. Anything it states
    must trace back to an observation, the goal, or an ASK result.
    Proper nouns are where factual harm concentrates, so those are checked.
    """
    grounded = goal.lower()
    for act, obs in history:
        grounded += " " + obs.lower()

    words = text.split()
    suspects = []
    for i, w in enumerate(words):
        bare = w.strip(".,!?'\"%")

        # Numbers are claims too. "1876" or "12 feet" stated without a
        # source is the same failure as an invented name, and it slips
        # through a capitalisation check because digits are not upper case.
        if bare and any(c.isdigit() for c in bare):
            if bare.lower() not in grounded:
                suspects.append(bare)
            continue

        if not bare or not bare[0].isupper():
            continue
        if i == 0 or words[i - 1].endswith((".", "!", "?")):
            continue                      # sentence-initial capitalisation
        if bare.lower() in ("i", "i'm", "i've"):
            continue
        if bare.lower() not in grounded:
            suspects.append(bare)
    return suspects


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
    plan = []
    target = goal_target(goal)
    nudged = False
    if verbose and target:
        print(f"[loop] search goal, target words: {sorted(target)}")

    for step in range(1, MAX_STEPS + 1):
        prompt = build_prompt(goal, history)

        if verbose:
            print(f"\n--- step {step} ---")

        if plan:
            # execute the rest of a plan the model already gave us
            action, arg = plan.pop(0)
            from_plan = True
            if verbose:
                print(f"plan:  {action}: {arg}".rstrip(": "))
        else:
            from_plan = False
            # after a rejection, raise temperature to break the token groove
            temp = 0.2 if not stuck_streak else min(0.2 + 0.35 * stuck_streak, 0.9)
            _t_llm = time.time()
            try:
                raw = ollama_generate(prompt, temperature=temp)
            except Exception as e:
                print(f"[loop] model call failed: {e}")
                hw.speak("my brain is not responding")
                break

            if verbose:
                print(f"model: {raw.strip()!r}")

            if TIMING:
                print(f"       [t] llm {time.time() - _t_llm:.1f}s")
            actions = parse_all(raw)
            if actions:
                action, arg = actions[0]
                # keep the rest, but only up to the first LOOK/ASK - anything
                # after an observation must be re-decided once we know what
                # the observation actually was. That is the whole point of ReAct.
                for a, g in actions[1:]:
                    plan.append((a, g))
                    if a in ("LOOK", "ASK"):
                        break   # anything after an observation must be re-decided
                if plan and verbose:
                    print(f"       (+{len(plan)} queued)")
            else:
                action, arg = None, "no action line found"

        # unparseable -> feed the error back, don't crash
        if action is None:
            note = "invalid, reply with ONE action line only"
            if verbose:
                print(f"  -> [rejected: {arg}] {note}")
            history.append((raw.strip()[:60] or "(empty)", note))
            stuck_streak += 1
            continue

        line = f"{action}: {arg}".strip().rstrip(":")
        # Qualify LOOK by heading: "left" from part 1 and "left" from part 3
        # are different places. Without this the robot forgets where it has
        # already searched every time it turns.
        # Key LOOKs by heading AND position. After a MOVE the robot is
        # somewhere new, so looking the same direction is a different view -
        # exactly the same reason TURN resets the heading component.
        key = (f"{line}@{hw.heading}#{hw.position}"
               if action == "LOOK" else line)

        # Loop breaker.
        # TURN is deliberately exempt: each turn changes the heading, so a
        # repeated TURN is a genuinely new physical state, not wasted work.
        # "turn left and turn left" is a legitimate goal. Turning is bounded
        # by the budget instead (turns_since_look cap + MAX_TURN_DEG).
        if key in seen_actions and action not in ("SAY", "TURN", "MOVE", "BACK"):
            repeats = seen_actions.count(key) + 1
            if action == "LOOK":
                _here = f"@{hw.heading}#{hw.position}"
                # Keys look like "LOOK: left down@0#0" - pull out the full
                # two-word direction, not just the first token, or every cell
                # reads as untried and the hint sends the model in circles.
                tried = {a.split(":", 1)[1].split("@")[0].strip()
                         for a in seen_actions
                         if a.startswith("LOOK") and a.endswith(_here)}
                # Fixed sweep: three heights at centre pan, then rotate.
                # Twelve views covers the room in four quarters.
                cells = ["center down", "center level", "center up"]
                untried = [c for c in cells if c not in tried]
                note = (f"that view is done. Still unchecked here: {untried[0]}."
                        if untried else
                        "every view here is done. Rotate to face a new part "
                        "of the room.")
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
            seen_actions.append(key)
            stuck_streak += 1
            plan = []          # plan is stale once something was rejected

            # Terminate on CONSECUTIVE failures only. Counting total repeats
            # of one key ends the run even when the model made real progress
            # between the duplicates - which is not being stuck.
            if stuck_streak >= 3:
                if verbose:
                    print("[loop] stuck - terminating and reporting")
                report(hw, history)
                finished = True
                break
            continue
        seen_actions.append(key)

        if action == "DONE":
            finished = True
            if verbose:
                print("[loop] agent signalled DONE")
            break

        if action == "SAY":
            bad = ungrounded_claims(arg, goal, history)
            if bad:
                note = ("that statement contains something you never observed: "
                        + ", ".join(bad[:3])
                        + ". Look it up before saying it, or leave it out.")
                if verbose:
                    print(f"  -> [grounding] blocked: {', '.join(bad[:3])}")
                history.append(("(note)", note))
                stuck_streak += 1
                plan = []
                if stuck_streak >= 3:
                    report(hw, history)
                    finished = True
                    break
                continue
            hw.speak(arg)
            history.append((f"SAY: {arg}", "spoken"))
            # The goal has been answered out loud. Waiting for the model to
            # emit DONE: is unreliable - it keeps searching instead. Once the
            # answer is grounded and spoken, the loop is over.
            if nudged:
                if verbose:
                    print("[loop] answer given - finishing")
                finished = True
                break
            continue

        searched = {AREA[int(a.split("@")[1].split("#")[0]) % 360]
                    for a in seen_actions
                    if a.startswith("LOOK") and "@" in a}
        obs = execute(hw, budget, action, arg, searched=searched,
                      planned=from_plan)
        # "nothing identifiable" is a real observation - it means this spot is
        # empty, which is exactly what a search needs to learn. Counting it as
        # a refusal ends the sweep after three empty looks.
        refused = obs.startswith(("refused", "invalid", "unknown", "cannot",
                                  "could not"))
        if refused:
            stuck_streak += 1
            plan = []          # plan is stale once something was refused
            if verbose:
                print(f"  -> {obs}")
            history.append((line, obs))
            if stuck_streak >= 3:
                if verbose:
                    print("[loop] stuck on refusals - terminating and reporting")
                report(hw, history)
                finished = True
                break
            continue
        stuck_streak = 0

        # Termination nudge: once the robot has real observations in hand,
        # a small model will often keep gathering instead of answering.
        # Remind it that it already has enough, without naming a verb.
        if action == "LOOK" and not obs.startswith(("refused", "invalid")):
            # Report goals are satisfied by any observation. Search goals are
            # satisfied only when the target actually appears - nudging early
            # on a search makes the robot give up before it has swept the room.
            if TIMING and target:
                print(f"       [target] {'MATCH' if target_found(target, obs) else 'no match'}"
                      f" for {sorted(target)}")
            if not nudged and (target is None or target_found(target, obs)):
                nudged = True
                history.append((line, obs))
                history.append(("(hint)",
                                "you now have enough information to answer the goal."))
                if verbose:
                    print(f"  -> {obs}")
                    print("  -> [nudge] enough information to answer")
                continue
        if verbose:
            print(f"  -> {obs}")
        history.append((line, obs))

        # Tell the model when it has rotated back somewhere already searched.
        if action == "TURN" and not obs.startswith("refused"):
            here = [a for a in seen_actions
                    if a.startswith("LOOK") and a.endswith(f"@{hw.heading}")]
            if len(here) >= 3:
                history.append(("(hint)",
                                f"area {AREA[hw.heading % 360]} was already "
                                "searched. Rotate again."))
                if verbose:
                    print("  -> [hint] already searched this part")

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
    ap.add_argument("--mute", action="store_true", help="skip TTS entirely")
    ap.add_argument("--model", default=MODEL_NAME)
    args = ap.parse_args()

    DRY_RUN = args.dry_run
    globals()["MODEL_NAME"] = args.model

    print(f"model: {MODEL_NAME} @ {AGENT_HOST}   goal: {args.goal!r}")
    hw = Hardware(dry_run=args.dry_run, mute=args.mute)

    t0 = time.time()
    history = react(args.goal, hw)
    print(f"\n[done] {len(history)} rounds in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
