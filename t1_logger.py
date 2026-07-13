"""
t1_logger.py
==============
Tier 1 component #1: the logging hook.

Import this in pipeline_text.py / voice_pipeline_routed.py and
call log_interaction() right after the robot answers. It appends
one JSON line to t1_log.jsonl and returns IMMEDIATELY - the kid
never waits on evaluation.

This is the "fire and forget" half of the async pattern:
the pipeline writes the triple, a separate process (t1_scorer.py)
picks it up later. On a real backend this file is replaced by a
Celery task queue - same shape, heavier plumbing.
"""

import json
import time
import os

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "t1_log.jsonl")


def log_interaction(question: str, contexts: list, answer: str, route: str):
    """Append one interaction. Costs ~1ms. Never blocks, never fails loudly."""
    try:
        entry = {
            "ts": time.time(),
            "question": question,
            "contexts": contexts if contexts else [],
            "answer": answer,
            "route": route,
            "scored": False,
        }
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass  # logging must never break the pipeline
