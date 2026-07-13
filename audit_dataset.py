# audit_dataset.py - quick scan
import json
rows = json.load(open("ragas_dataset.json"))

print(f"Total: {len(rows)}")
print(f"\nRouting distribution:")
from collections import Counter
print(Counter(r["classified_as"] for r in rows))
print(f"\nExpected vs classified mismatches:")
for r in rows:
    exp, got = r["expected_source"], r["classified_as"]
    if (exp == "knowledge_base" and got == "LIVE_DATA") or \
       (exp in ("news","weather") and got != "LIVE_DATA"):
        print(f"  [{exp} -> {got}] {r['question'][:55]}")
print(f"\nRows with empty contexts: {sum(1 for r in rows if not r['contexts'])}")
print(f"[VERIFY] ground truths to check: {sum(1 for r in rows if '[VERIFY]' in r['ground_truth'])}")
