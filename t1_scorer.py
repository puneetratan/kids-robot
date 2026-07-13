"""
t1_scorer.py
==============
Tier 1 component #2: the background scorer.

Runs as a SEPARATE process (the Pi's "Celery worker"). Tails
t1_log.jsonl, finds unscored interactions, scores each with the
two REFERENCE-FREE metrics - the exact mechanisms we built by
hand in Videos 11-12:

  faithfulness      (claims from answer, verified vs contexts)
  answer_relevancy  (reverse-questions, embedded, averaged)

No ground truth needed - that's what makes T1 possible on live
traffic. Scores are written to t1_scores.jsonl for the dashboard.

Run in its own terminal:
    export ANTHROPIC_API_KEY=your_key
    python3 t1_scorer.py

Notes:
- STATIC-route rows are skipped for faithfulness (no context by
  design) but still get relevancy.
- SAMPLE_RATE controls cost: 1.0 scores everything (demo),
  production would use 0.05-0.10.
"""

import json
import time
import os
import random

# Reuse the manual implementations - the whole point of building them!
from faithfulness_manual import faithfulness
from answer_relevancy_manual import answer_relevancy

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "t1_log.jsonl")
SCORES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "t1_scores.jsonl")
POLL_SECONDS = 5
SAMPLE_RATE = 1.0   # demo: score everything. production: 0.05-0.10

FAITHFULNESS_ALERT = 0.5   # thresholds calibrated from our golden-set runs
RELEVANCY_ALERT = 0.45     # (mxbai similarity texture - deflections ~0.32)


def load_scored_timestamps():
    done = set()
    if os.path.exists(SCORES_FILE):
        with open(SCORES_FILE) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["ts"])
                except Exception:
                    continue
    return done


def score_entry(entry):
    q, ans, ctxs = entry["question"], entry["answer"], entry["contexts"]

    result = {
        "ts": entry["ts"],
        "question": q,
        "route": entry["route"],
        "faithfulness": None,
        "answer_relevancy": None,
        "alert": False,
    }

    # relevancy: always computable (needs only Q + A)
    try:
        result["answer_relevancy"] = round(
            answer_relevancy(q, ans, verbose=False), 3)
    except Exception as e:
        print(f"  [relevancy error] {e}")

    # faithfulness: only when contexts exist (STATIC rows skip by design)
    if ctxs:
        try:
            context_text = "\n".join(ctxs)
            result["faithfulness"] = round(
                faithfulness(q, ans, context_text, verbose=False), 3)
        except Exception as e:
            print(f"  [faithfulness error] {e}")

    f_bad = result["faithfulness"] is not None and result["faithfulness"] < FAITHFULNESS_ALERT
    r_bad = result["answer_relevancy"] is not None and result["answer_relevancy"] < RELEVANCY_ALERT
    result["alert"] = f_bad or r_bad
    return result


def main():
    print("T1 Scorer running - watching for new interactions...")
    print(f"(sample rate {SAMPLE_RATE:.0%}, poll every {POLL_SECONDS}s, Ctrl+C to stop)\n")

    while True:
        scored = load_scored_timestamps()
        pending = []
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE) as f:
                for line in f:
                    try:
                        e = json.loads(line)
                        if e["ts"] not in scored:
                            pending.append(e)
                    except Exception:
                        continue

        for entry in pending:
            if random.random() > SAMPLE_RATE:
                # mark sampled-out entries as seen with null scores
                with open(SCORES_FILE, "a") as f:
                    f.write(json.dumps({"ts": entry["ts"], "sampled_out": True}) + "\n")
                continue

            print(f"Scoring: {entry['question'][:55]}")
            result = score_entry(entry)
            with open(SCORES_FILE, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            flag = " 🚨 ALERT" if result["alert"] else ""
            print(f"  faithfulness={result['faithfulness']}  "
                  f"relevancy={result['answer_relevancy']}{flag}\n")

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
