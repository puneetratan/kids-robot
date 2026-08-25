"""
generate_v4_data.py
=====================
Fixes the VISION class and adds speed commands.

WHAT WENT WRONG IN V3
---------------------
Every vision phrasing routed to ACTION. "What do you see?" -
verbatim in the training data - classified as a movement
command with a 0.76 margin. Confidently wrong, not uncertain.

The cause was class imbalance, not a mysterious bug: ~80 ACTION
examples against ~30 VISION ones. Short imperative phrases
pattern-match to commands, and with ACTION dominating, the
model learned that shape means "move".

THE FIX
-------
Three things:

1. VISION gets roughly as many examples as ACTION. Balance
   alone does a lot of the work.

2. Near-miss pairs. This is the important part. Generic
   examples do not teach a boundary - contrasting ones do:

       "look at this"     VISION    vs  "look left"   VISION
       "what do you see"  VISION    vs  "what are you doing"
       "come and see"     VISION    vs  "come here"   ACTION
       "turn left"        ACTION    vs  "look left"   VISION

   The collisions are where the model failed, so the collisions
   are what it needs to see.

3. "look left" is VISION, not ACTION. Direction becomes a
   parameter of looking rather than a separate class - the
   handler pans the neck, then captures, then describes. That
   keeps one meaning of "left" for the wheels and another for
   the head, without a third class to confuse things.

Also adds speed commands to ACTION, which did not exist before.

Usage:
    python3 generate_v4_data.py
    -> reads classifier_training_data.json
    -> writes classifier_training_data_v4.json
"""

import json
import os

SRC = "classifier_training_data.json"
OUT = "classifier_training_data_v4.json"


# ============================================================
# VISION - questions answered from the camera.
# Deliberately as large as ACTION. The imbalance was the bug.
# ============================================================
VISION = [
    # the plain question forms that failed in v3
    "what do you see", "what can you see", "what do you see now",
    "what is this", "what's this", "what is this thing",
    "what is that", "what's that", "what is this object",
    "what am I holding", "what am I showing you",
    "what is in my hand", "what's in my hand",

    # imperatives
    "look", "look at this", "look here", "look at that",
    "take a look", "have a look", "look at my hand",
    "see this", "see what I have", "check this out",
    "look at what I have", "look at my toy",

    # directional looking - the neck moves, then it describes
    "look left", "look to the left", "look at your left",
    "look right", "look to the right", "look at your right",
    "look up", "look at the ceiling", "look above you",
    "look down", "look at the floor", "look below",
    "look around", "look behind that", "turn your head and look",
    "look over there", "what's on your left", "what's on your right",
    "what do you see on the left", "what do you see up there",

    # describe
    "describe what you see", "describe this", "describe that",
    "tell me what you see", "tell me what this is",
    "can you see this", "can you see my toy", "do you see this",
    "can you see anything", "what does this look like",

    # properties of what is in frame
    "what colour is this", "what color is this",
    "what colour is my shirt", "what color is that",
    "how many things do you see", "is this red",
    "what shape is this", "is it big or small",

    # capture
    "take a picture", "take a photo", "snap a picture",
    "click a photo", "capture this", "photograph this",

    # kid phrasings
    "look what I got", "see my drawing", "look at my picture",
    "what animal is this", "is this a dinosaur",
    "guess what I'm holding", "can you guess this",
    "what toy is this", "who is this", "what's in front of you",
]


# ============================================================
# ACTION - commands. Speed is new.
# ============================================================
ACTION = [
    # forward
    "move forward", "go forward", "go ahead", "drive forward",
    "move ahead", "go straight", "drive straight ahead",
    "start moving", "start driving", "keep going",
    "go", "move", "come here", "come to me", "go that way",
    "walk", "drive", "forward", "go go go",

    # backward
    "come back", "go back", "move back", "back up",
    "reverse", "go backward", "move backward", "back",
    "come backwards", "go the other way", "retreat",

    # left / right - the WHEELS, not the head
    "turn left", "go left", "move left", "left",
    "turn to the left", "take a left", "spin left",
    "drive left", "steer left",
    "turn right", "go right", "move right", "right",
    "turn to the right", "take a right", "spin right",
    "drive right", "steer right",

    # stop
    "stop", "stop right there", "stop moving", "halt",
    "wait", "freeze", "hold on", "stop now", "don't move",
    "stay there", "stop please", "that's enough",

    # speed - new in v4
    "go faster", "faster", "speed up", "go quicker",
    "move faster", "drive faster", "pick up speed",
    "go slower", "slower", "slow down", "go slowly",
    "move slowly", "drive slowly", "take it slow",
    "be careful", "go gently", "not so fast",

    # misc
    "turn around", "spin around", "drive to the door",
    "come towards me", "move a little", "go a bit further",
    "keep driving", "start", "resume",
]


# ============================================================
# Boundary pairs. The whole point of this retrain.
#
# Each pair is two phrasings that differ by one word and mean
# entirely different things. Without these the model learns
# "short imperative = ACTION" and swallows VISION whole.
# ============================================================
BOUNDARY = [
    # look = head + describe. turn/go = wheels.
    ("look left", "VISION"),
    ("turn left", "ACTION"),
    ("look right", "VISION"),
    ("turn right", "ACTION"),
    ("look up", "VISION"),
    ("go up", "ACTION"),
    ("look down", "VISION"),
    ("look around", "VISION"),
    ("turn around", "ACTION"),
    ("look forward", "VISION"),
    ("move forward", "ACTION"),
    ("look back", "VISION"),
    ("go back", "ACTION"),

    # see vs come
    ("come and see", "VISION"),
    ("come here", "ACTION"),
    ("see this", "VISION"),
    ("go there", "ACTION"),

    # what-questions that are not vision
    ("what do you see", "VISION"),
    ("what are you doing", "STATIC"),
    ("what can you do", "STATIC"),
    ("how fast can you go", "STATIC"),
    ("how do your wheels work", "STATIC"),
    ("how does your camera work", "STATIC"),
    ("can robots see", "STATIC"),
    ("how do eyes work", "STATIC"),
    ("why do we have two eyes", "STATIC"),

    # polite commands are still commands
    ("can you move forward", "ACTION"),
    ("could you stop", "ACTION"),
    ("will you turn left", "ACTION"),
    ("can you go faster", "ACTION"),

    # polite vision requests are still vision
    ("can you look at this", "VISION"),
    ("could you tell me what this is", "VISION"),
    ("will you look at my toy", "VISION"),
]


# ============================================================
# Misroutes found by RAGAS evaluation, carried over from v3.
# ============================================================
ADVERSARIAL = [
    # arithmetic is STATIC - "today" must not pull it to LIVE_DATA
    ("what is 7 times 8", "STATIC"),
    ("what is 7 times 8 today", "STATIC"),
    ("what's 12 plus 15", "STATIC"),
    ("how much is 9 times 6", "STATIC"),
    ("what is 100 divided by 4", "STATIC"),
    ("what is 6 times 7 right now", "STATIC"),

    # settled facts containing a future year -> KNOWLEDGE_BASE
    ("which countries are hosting the 2026 World Cup", "KNOWLEDGE_BASE"),
    ("how many cities will host games in the 2026 World Cup", "KNOWLEDGE_BASE"),
    ("where is the 2026 World Cup being held", "KNOWLEDGE_BASE"),
    ("how many teams play in the 2026 World Cup", "KNOWLEDGE_BASE"),

    # genuinely live -> LIVE_DATA
    ("who won the World Cup", "LIVE_DATA"),
    ("who won the FIFA World Cup 2026", "LIVE_DATA"),
    ("what was the score in yesterday's match", "LIVE_DATA"),
    ("who won the game last night", "LIVE_DATA"),
    ("what's happening in the World Cup today", "LIVE_DATA"),
    ("who reached the semifinals", "LIVE_DATA"),

    # Ramayana phrasings all land the same way
    ("who helped Rama rescue Sita", "KNOWLEDGE_BASE"),
    ("who saved Sita from Ravana", "KNOWLEDGE_BASE"),
    ("who rescued Sita", "KNOWLEDGE_BASE"),
    ("tell me about Hanuman", "KNOWLEDGE_BASE"),
    ("who is the main hero in the Ramayana", "KNOWLEDGE_BASE"),
    ("who fought Ravana", "KNOWLEDGE_BASE"),
]


def main():
    if not os.path.exists(SRC):
        raise SystemExit(f"{SRC} not found - run from the repo root")

    data = json.load(open(SRC))
    original = len(data)

    before = {}
    for label in (d["label"] for d in data):
        before[label] = before.get(label, 0) + 1

    for text in VISION:
        data.append({"text": text, "label": "VISION"})
    for text in ACTION:
        data.append({"text": text, "label": "ACTION"})
    for text, label in BOUNDARY + ADVERSARIAL:
        data.append({"text": text, "label": label})

    seen, deduped = set(), []
    for row in data:
        key = row["text"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    with open(OUT, "w") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)

    after = {}
    for label in (d["label"] for d in deduped):
        after[label] = after.get(label, 0) + 1

    print(f"{SRC}: {original} examples")
    for label, n in sorted(before.items()):
        print(f"    {label:<16} {n}")

    print(f"\n{OUT}: {len(deduped)} examples")
    for label, n in sorted(after.items()):
        print(f"    {label:<16} {n}")

    gap = abs(after.get("ACTION", 0) - after.get("VISION", 0))
    print(f"\nACTION/VISION gap: {gap}")
    if gap > 20:
        print("  still lopsided - the smaller class will lose ties")
    else:
        print("  balanced enough that neither should swallow the other")

    print("""
Next:
    upload to Colab, load classifier_training_data_v4.json,
    keep the 5-class label map, retrain, download
    distilbert_classifier_final/ back to the Pi.

Then check the boundary pairs before wiring anything up:

    python3 -c "
    from shiva import classify
    for q in ['look left', 'turn left', 'what do you see',
              'what are you doing', 'go faster', 'come and see',
              'come here', 'take a picture', 'how fast can you go']:
        print(f'{classify(q)[0]:<16} {q}')
    "

Every 'look' should be VISION, every 'turn'/'go' ACTION.
""")


if __name__ == "__main__":
    main()
