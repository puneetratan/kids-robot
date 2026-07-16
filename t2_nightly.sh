#!/bin/bash
# t2_nightly.sh - Tier 2: nightly golden-set evaluation
# Re-runs the frozen golden dataset through TODAY's pipeline and
# appends dated scores to t2_history.csv - "the grade book is forever."
#
# Install as cron (2am nightly):
#   crontab -e
#   0 2 * * * /home/pi/robot/t2_nightly.sh >> /home/pi/robot/t2_cron.log 2>&1
#
# Requires: MCP server running, ANTHROPIC_API_KEY set below.

cd /home/pi/robot
source venv/bin/activate
export ANTHROPIC_API_KEY="your_key_here"

# 1. Fresh pipeline outputs for the FROZEN golden questions
python3 generate_ragas_dataset.py

# 2. 4-metric eval
python3 run_ragas_eval.py

# 3. Append dated summary to history
python3 - << 'PYEOF'
import csv, os
from datetime import date
import pandas as pd

if os.path.exists("ragas_results.csv"):
    df = pd.read_csv("ragas_results.csv")
    row = {
        "date": str(date.today()),
        "n": len(df),
        "faithfulness": round(df["faithfulness"].mean(), 3),
        "answer_relevancy": round(df["answer_relevancy"].mean(), 3),
        "context_recall": round(df["context_recall"].mean(), 3),
        "context_precision": round(df["context_precision"].mean(), 3),
    }
    new = not os.path.exists("t2_history.csv")
    with open("t2_history.csv", "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if new: w.writeheader()
        w.writerow(row)
    print(f"T2 {row['date']}: f={row['faithfulness']} recall={row['context_recall']}")
PYEOF
