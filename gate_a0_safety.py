"""
gate_a0_safety.py
===================
Video 14: Gate A0 - the CHILD SAFETY gate.

Runs FIRST, before classification, before A1-A3/B1-B2.
Different axis entirely from the hallucination gates:

  A1-A3, B1-B2:  "is this answer TRUE/GROUNDED?"     (quality)
  A0 (this file): "is this topic APPROPRIATE for a
                  child, regardless of whether we
                  could answer it accurately?"        (safety)

Design choices that DELIBERATELY differ from the other gates:
  - Uses a judge LLM (Claude), not string/regex checks.
    Keyword blocklists are fragile - kids phrase things
    obliquely, and false negatives matter more here than
    anywhere else in the pipeline. ~200-400ms cost is
    ACCEPTED for reliability, unlike the millisecond-budget
    quality gates.
  - The fallback is a WARM REDIRECT to a trusted adult, not
    a generic "I don't know" - tone matters at these stakes.
  - Every A0 trigger is LOGGED SEPARATELY (a0_events.jsonl)
    so a parent can see WHAT KIND of question came up and
    HOW OFTEN - visibility as a feature, not a hidden event.
    The question text itself is logged so a parent can review
    it; treat this log file with the same care as any other
    record of a child's activity in your home.

Setup:
    export ANTHROPIC_API_KEY=your_key

Usage (as a library, called from the pipeline BEFORE classify):
    from gate_a0_safety import check_child_safety
    safe, category, redirect = check_child_safety(question)
    if not safe:
        speak(redirect)
        return   # never reaches the classifier or generation
"""

import json
import os
import time
import anthropic

client = anthropic.Anthropic()
JUDGE_MODEL = "claude-sonnet-4-5"

EVENTS_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a0_events.jsonl")

CATEGORIES = [
    "SAFE",              # normal kid question - proceed as usual
    "SEXUAL_CONTENT",
    "VIOLENCE_OR_WEAPONS",
    "SELF_HARM_OR_SAFETY",
    "SUBSTANCES",
    "OTHER_ADULT_TOPIC",  # death, family conflict, scary news, etc.
]

REDIRECT_MESSAGES = {
    "SEXUAL_CONTENT":
        "That's a really good question to ask a parent or grown-up you trust — "
        "they can explain it best! Want to ask me something else? 🤖",
    "VIOLENCE_OR_WEAPONS":
        "That sounds like something to talk about with a parent or teacher. "
        "I'm better at fun questions — want to try one? 🤖",
    "SELF_HARM_OR_SAFETY":
        "That's really important, and I think a parent, teacher, or trusted "
        "grown-up would be the best person to talk to about that. "
        "They care about you a lot! 💛",
    "SUBSTANCES":
        "That's a question for a parent or grown-up to answer. "
        "Let's find something else to explore together! 🤖",
    "OTHER_ADULT_TOPIC":
        "That's a big question! I think your parent or a grown-up you trust "
        "could explain that better than me. Want to ask me something else? 🤖",
}

DEFAULT_REDIRECT = REDIRECT_MESSAGES["OTHER_ADULT_TOPIC"]


def _classify_safety(question: str) -> str:
    """Single judge call. Returns one of CATEGORIES."""
    prompt = f"""You are a safety classifier for a voice assistant used by children
(ages roughly 5-10). Classify the following question into EXACTLY ONE category.

Categories:
SAFE - a normal question a child would ask (science, stories, games, weather,
       animals, math, general curiosity - even if oddly phrased)
SEXUAL_CONTENT - anything about sex, bodies in a sexual context, reproduction
       beyond basic age-appropriate biology
VIOLENCE_OR_WEAPONS - questions seeking instructions or details about weapons,
       violence, or how to hurt someone/something
SELF_HARM_OR_SAFETY - anything suggesting the child or someone else may be in
       danger, hurting themselves, or unsafe at home
SUBSTANCES - drugs, alcohol, smoking
OTHER_ADULT_TOPIC - death/grief, serious family conflict, frightening news
       events, or other topics better handled by a trusted adult

Be conservative: a normal curious kid question about how animals reproduce,
where babies come from in a basic biological sense, or historical violence
in an educational context (e.g. "what happened in World War 2") should
usually be SAFE unless graphic or explicit. When genuinely unsure between
SAFE and another category, prefer the SAFER (non-SAFE) classification.

Question: {question}

Respond with ONLY the category name, nothing else."""

    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=20,
        messages=[{"role": "user", "content": prompt}],
    )
    result = response.content[0].text.strip().upper()
    for cat in CATEGORIES:
        if cat in result:
            return cat
    return "OTHER_ADULT_TOPIC"  # unrecognized judge output -> fail safe, not open


def _log_event(question: str, category: str):
    try:
        with open(EVENTS_LOG, "a") as f:
            f.write(json.dumps({
                "ts": time.time(),
                "question": question,
                "category": category,
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass  # logging must never break the gate


def check_child_safety(question: str, verbose: bool = True):
    """
    Returns (safe: bool, category: str, redirect_message: str | None)

    safe=True  -> category is 'SAFE', redirect_message is None,
                  pipeline proceeds to classify_query() as normal
    safe=False -> category names WHY, redirect_message is what to
                  speak instead. Pipeline should log_interaction()
                  with route='GATED-A0' and RETURN immediately -
                  never reach the classifier or generation.
    """
    try:
        category = _classify_safety(question)
    except Exception as e:
        if verbose:
            print(f"  [GATE-A0 error] {e} - failing SAFE-BUT-LOGGED is wrong here;")
            print(f"  failing CLOSED (redirect) is the safer default on error.")
        _log_event(question, "JUDGE_ERROR")
        return False, "JUDGE_ERROR", DEFAULT_REDIRECT

    if category == "SAFE":
        if verbose:
            print(f"  [GATE-A0] safe - proceeding")
        return True, "SAFE", None

    _log_event(question, category)
    redirect = REDIRECT_MESSAGES.get(category, DEFAULT_REDIRECT)
    if verbose:
        print(f"  🛑 [GATE-A0] {category} - redirecting to trusted adult")
    return False, category, redirect
