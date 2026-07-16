"""
confidence_layer.py
=====================
Video 14: the Confidence & Fallback layer - protecting the answer
BEFORE it is spoken. Milliseconds, no judge LLM in the loop.

Two checkpoints, informed directly by our RAGAS findings:

CHECKPOINT A - PRE-GENERATION (~free, catches the predictable):
  A1. Classifier confidence: softmax margin on DistilBERT.
      Catches: "What is 7 times 8" -> LIVE_DATA (worst-10 case!)
  A2. Retrieval strength: were ChromaDB distances weak?
      Catches: the Hanuman disease (KB-routed, nothing retrieved)
  A3. Context thinness: is the live/KB context too short or
      question-irrelevant to ground an answer?
      Catches: broad news queries ("what's happening in sports")
      - 5 of our worst-10.

CHECKPOINT B - POST-GENERATION, PRE-SPEECH (string-level, no LLM):
  B1. Entity leakage: proper-noun-ish tokens in the answer that
      appear nowhere in the context (memory-gambling signature).
      Catches: Senegal/Belgium, Dallas/Houston inventions.
  B2. Date sanity: past dates claimed for future/current events.
      Catches: the "June 24, 2026" phantom-date hallucination.

Every check returns (ok: bool, reason: str). The pipeline decides:
  A fails -> don't generate; honesty fallback (or re-route)
  B fails -> suppress the draft; honesty fallback

Thresholds are CALIBRATED FROM OUR DATA, commented inline.
"""

import re
try:
    import torch          # needed only for check_classifier (A1)
except ImportError:       # string-level gates work without it
    torch = None

# ============================================================
# CHECKPOINT A
# ============================================================

def classifier_confidence(logits) -> tuple[float, float]:
    """Return (top_prob, margin). Margin = top1 - top2 softmax prob.
    Our misroutes ("7 times 8", "who save Sita") showed low margins;
    clean routes exceed ~0.6 margin. Calibrate on your test set."""
    probs = torch.softmax(logits, dim=1)[0]
    top2 = torch.topk(probs, 2).values
    return float(top2[0]), float(top2[0] - top2[1])


def check_classifier(logits, min_top=0.70, min_margin=0.30):
    top, margin = classifier_confidence(logits)
    ok = top >= min_top and margin >= min_margin
    return ok, f"classifier top={top:.2f} margin={margin:.2f}" + ("" if ok else " LOW")


def check_retrieval(distances, max_distance=0.6):
    """A2: KB route - did anything actually pass the threshold?
    Empty = the Hanuman disease. We also flag 'barely passed'
    (all distances in the top 15% of the allowed band)."""
    if not distances:
        return False, "retrieval EMPTY (Hanuman case)"
    best = min(distances)
    if best > max_distance * 0.85:
        return False, f"retrieval WEAK (best distance {best:.3f})"
    return True, f"retrieval ok (best {best:.3f})"


STOPWORDS = set("""a an the is are was were be been what who when where why how
which do does did can could will would of in on at to for with about and or""".split())

def check_context_thinness(question: str, context: str,
                           min_chars=60, min_overlap=1):
    """A3: is the context substantial AND topically related?
    Overlap = shared non-stopword terms between Q and context.
    min_chars=60 calibrated so real weather forecasts (~65-90
    chars) pass while empty/one-liner contexts fail. Broad-news
    failures shared ~0 content words with the question."""
    if not context or len(context.strip()) < min_chars:
        return False, f"context THIN ({len(context.strip()) if context else 0} chars)"
    q_terms = {w for w in re.findall(r"[a-z']+", question.lower()) if w not in STOPWORDS}
    c_lower = context.lower()
    overlap = sum(1 for t in q_terms if t in c_lower)
    if overlap < min_overlap:
        return False, f"context OFF-TOPIC (0/{len(q_terms)} question terms present)"
    return True, f"context ok ({overlap}/{len(q_terms)} terms overlap)"


# ============================================================
# CHECKPOINT B
# ============================================================

def _proper_nouns(text: str) -> set[str]:
    """Capitalized tokens not at sentence start - cheap NER."""
    tokens = re.findall(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,})\b", " " + text)
    return {t for t in tokens if t.lower() not in STOPWORDS}


def check_entity_leakage(answer: str, context: str, max_leaked=1):
    """B1: proper nouns in the answer absent from the context.
    Senegal/Belgium leaked 2, Dallas/Houston leaked 2 - so the
    threshold is max_leaked=1: one leak tolerated (analogies can
    legitimately introduce a single name), two+ = gambling."""
    if not context:
        return True, "no context (STATIC) - skip"
    ans_ents = _proper_nouns(answer)
    ctx_lower = context.lower()
    leaked = {e for e in ans_ents if e.lower() not in ctx_lower}
    ok = len(leaked) <= max_leaked
    return ok, f"entities leaked: {sorted(leaked)[:4]}" if not ok else f"entities ok ({len(leaked)} leaked)"


MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"

def check_date_sanity(answer: str, today):
    """B2: the phantom-date check. If the answer claims a FUTURE
    meeting/match 'will happen' on a date already in the past
    (like 'June 24, 2026' said in July), flag it."""
    for m in re.finditer(rf"({MONTHS})\s+(\d{{1,2}}),?\s+(\d{{4}})", answer, re.I):
        month_name, day, year = m.group(1), int(m.group(2)), int(m.group(3))
        month = ("january february march april may june july august "
                 "september october november december").split().index(month_name.lower()) + 1
        try:
            from datetime import date
            claimed = date(year, month, day)
            if claimed < today and re.search(r"\bwill\b|\bgoing to\b|upcoming", answer, re.I):
                return False, f"date INSANE: 'will happen' on past date {claimed}"
        except ValueError:
            return False, f"date INVALID: {m.group(0)}"
    return True, "dates ok"


# ============================================================
# Orchestration helpers
# ============================================================

FALLBACK = ("Hmm, I'm not confident I know that one! "
            "Let me not guess - ask me something else, or teach me! 🤖")


def checkpoint_a(logits=None, distances=None, context=None, question=None,
                 route="STATIC", verbose=True):
    """Run applicable pre-generation checks. Returns (proceed, reasons)."""
    reasons = []
    ok_all = True

    if logits is not None:
        ok, r = check_classifier(logits)
        reasons.append("A1 " + r); ok_all &= ok

    if route == "KNOWLEDGE_BASE":
        ok, r = check_retrieval(distances or [])
        reasons.append("A2 " + r); ok_all &= ok

    if route in ("KNOWLEDGE_BASE", "LIVE_DATA") and context is not None:
        ok, r = check_context_thinness(question or "", context)
        reasons.append("A3 " + r); ok_all &= ok

    if verbose:
        for r in reasons:
            print(f"  [GATE-A] {r}")
    return ok_all, reasons


def checkpoint_b(answer, context, today=None, verbose=True):
    """Run post-generation sanity checks. Returns (speak, reasons)."""
    from datetime import date
    today = today or date.today()
    reasons = []
    ok_all = True

    ok, r = check_entity_leakage(answer, context)
    reasons.append("B1 " + r); ok_all &= ok

    ok, r = check_date_sanity(answer, today)
    reasons.append("B2 " + r); ok_all &= ok

    if verbose:
        for r in reasons:
            print(f"  [GATE-B] {r}")
    return ok_all, reasons
