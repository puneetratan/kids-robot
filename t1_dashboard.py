"""
t1_dashboard.py
=================
Tier 1 component #3: the live dashboard.

Run in a third terminal. Refreshes every few seconds, showing
rolling quality stats and recent alerts - the "Grafana" of the
Pi, in 80 lines of terminal output.

    python3 t1_dashboard.py
"""

import json
import os
import time

SCORES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "t1_scores.jsonl")
REFRESH = 5


def load_scores():
    rows = []
    if os.path.exists(SCORES_FILE):
        with open(SCORES_FILE) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if not r.get("sampled_out"):
                        rows.append(r)
                except Exception:
                    continue
    return rows


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def fmt(v):
    return f"{v:.2f}" if v is not None else " -- "


def main():
    while True:
        os.system("clear")
        rows = load_scores()

        print("=" * 62)
        print("  T1 LIVE EVALUATION DASHBOARD".center(62))
        print("=" * 62)

        if not rows:
            print("\n  waiting for scored interactions...\n")
        else:
            f_avg = mean([r["faithfulness"] for r in rows])
            r_avg = mean([r["answer_relevancy"] for r in rows])
            alerts = [r for r in rows if r.get("alert")]

            print(f"\n  interactions scored : {len(rows)}")
            print(f"  avg faithfulness    : {fmt(f_avg)}")
            print(f"  avg relevancy       : {fmt(r_avg)}")
            print(f"  alerts              : {len(alerts)} "
                  f"({len(alerts)/len(rows):.0%} of traffic)")

            print("\n  " + "-" * 58)
            print("  LAST 8 INTERACTIONS")
            print("  " + "-" * 58)
            for r in rows[-8:]:
                flag = "🚨" if r.get("alert") else "  "
                print(f"  {flag} f={fmt(r['faithfulness'])} "
                      f"r={fmt(r['answer_relevancy'])}  "
                      f"[{r['route'][:4]}] {r['question'][:34]}")

            if alerts:
                print("\n  " + "-" * 58)
                print("  RECENT ALERTS (likely hallucinations/deflections)")
                print("  " + "-" * 58)
                for r in alerts[-4:]:
                    print(f"  🚨 f={fmt(r['faithfulness'])} "
                          f"r={fmt(r['answer_relevancy'])}  {r['question'][:40]}")

        print(f"\n  refreshing every {REFRESH}s - Ctrl+C to stop")
        time.sleep(REFRESH)


if __name__ == "__main__":
    main()
