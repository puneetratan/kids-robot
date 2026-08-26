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
LOOK: <direction>             - point the camera and describe what is there\n                                (direction is one of: left, center, right)
TURN: <direction>             - rotate the robot 90 degrees\n                                (direction is one of: left, right)
ASK: <question>               - look up a fact about the world
SAY: <text>                   - speak the answer to the human
DONE:                         - stop

The room has four areas: A, B, C, D. One TURN moves you to the next area.
Two turns in the same direction put you in the area behind you.

Rules:
- You may only report things that appeared in a "->" line above. Never guess.
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

Example - searching for something:
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

Example - looking up a fact:
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
        if action not in ("(note)", "(hint)"):
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
        self.turns_since_look = 0

    def allow_ask(self):
        if self.asks >= 3:
            return False, "refused: no more lookups. Report what you already know."
        self.asks += 1
        return True, None

    def allow_move(self, feet):
        if feet <= 0 or feet > 6:
            return False, f"refused: single move must be 1-6 feet"
        if self.distance + feet > MAX_DISTANCE:
            return False, f"refused: distance budget exhausted"
        self.distance += feet
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
        # Reject menu echoes like "left | center | right" and multi-word args.
        # Silently taking the first token means acting on a command the model
        # never actually chose.
        if "|" in d or "," in d or len(d.split()) > 1:
            return ("invalid: choose exactly one direction, not a list")
        if d not in ("left", "center", "right"):
            return "invalid: the direction must be left, center or right"
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
    """True if the observation plausibly contains the searched-for thing."""
    if not target:
        return False
    seen = {w.strip(".,?!").lower() for w in obs.split()}
    return bool(target & seen)


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
        bare = w.strip(".,!?'\"")
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
            try:
                raw = ollama_generate(prompt, temperature=temp)
            except Exception as e:
                print(f"[loop] model call failed: {e}")
                hw.speak("my brain is not responding")
                break

            if verbose:
                print(f"model: {raw.strip()!r}")

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
        key = f"{line}@{hw.heading}" if action == "LOOK" else line

        # Loop breaker.
        # TURN is deliberately exempt: each turn changes the heading, so a
        # repeated TURN is a genuinely new physical state, not wasted work.
        # "turn left and turn left" is a legitimate goal. Turning is bounded
        # by the budget instead (turns_since_look cap + MAX_TURN_DEG).
        if key in seen_actions and action not in ("SAY", "TURN"):
            repeats = seen_actions.count(key) + 1
            if action == "LOOK":
                tried = {a.split(":")[1].split("@")[0].strip()
                         for a in seen_actions
                         if a.startswith("LOOK") and a.endswith(f"@{hw.heading}")}
                untried = [d for d in ("left", "center", "right") if d not in tried]
                note = (f"that direction is done. Only the {untried[0]} side "
                        "is still unchecked."
                        if untried else
                        "all three directions here are done. Rotate to face a "
                        "new part of the room.")
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

            # give up on steering and terminate with whatever was learned
            if repeats >= 3:
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
            continue

        searched = {AREA[int(a.split("@")[1]) % 360] for a in seen_actions
                    if a.startswith("LOOK") and "@" in a}
        obs = execute(hw, budget, action, arg, searched=searched,
                      planned=from_plan)
        refused = obs.startswith(("refused", "invalid", "unknown", "cannot"))
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