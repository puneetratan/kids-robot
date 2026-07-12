"""
run_ragas_eval.py
===================
Step 3 of Video 11: run the RAGAS library on ragas_dataset.json
with Claude as the LLM judge.

This does at scale what faithfulness_manual.py did by hand —
plus 3 more metrics:

  faithfulness       : answer grounded in retrieved context?
                       (claim extraction + verification — the
                        exact mechanism we built manually)
  answer_relevancy   : does the answer address the question?
  context_recall     : did retrieval fetch what was needed to
                       produce the ground truth?
  context_precision  : were the retrieved chunks useful or noise?

Judge LLM  : Claude (via langchain-anthropic)
Embeddings : mxbai-embed-large via local Ollama (free, no API cost —
             answer_relevancy needs embeddings)

Setup (in venv):
    pip install ragas langchain-anthropic langchain-ollama datasets
    export ANTHROPIC_API_KEY=your_key

Run:
    python3 run_ragas_eval.py
"""

import json
import numpy as np
from datasets import Dataset

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

from ragas import evaluate, RunConfig
from langchain_anthropic import ChatAnthropic
from langchain_ollama import OllamaEmbeddings

# ============================================================
# Judge + embeddings setup
# ============================================================
print("Setting up Claude as judge...")
judge_llm = LangchainLLMWrapper(
    ChatAnthropic(model="claude-sonnet-4-5", temperature=0, max_tokens=2000)
)

print("Setting up local Ollama embeddings...")
local_embeddings = LangchainEmbeddingsWrapper(
    OllamaEmbeddings(model="mxbai-embed-large")
)

# ============================================================
# Load dataset
# ============================================================
with open("ragas_dataset.json") as f:
    rows = json.load(f)

# Filter out pipeline errors and weather rows with runtime ground truth
eval_rows = [
    r for r in rows
    if r["classified_as"] != "ERROR"
    and r["ground_truth"] != "WEATHER_LIVE"
    and r["contexts"]          # RAGAS needs at least one context
]
skipped = len(rows) - len(eval_rows)
print(f"Loaded {len(rows)} rows, evaluating {len(eval_rows)} (skipped {skipped}: errors/weather-live/no-context)")

# ============================================================
# IMPORTANT: audit the skips before they hide failures.
# Empty contexts mean different things depending on the route:
#   STATIC          -> empty BY DESIGN (no retrieval attempted) - fine
#   KNOWLEDGE_BASE  -> RETRIEVAL FAILURE (doc exists or should,
#                      nothing passed the distance threshold)
#   LIVE_DATA       -> MCP FETCH FAILURE (tool errored or empty)
# These are effectively context_recall = 0 rows that RAGAS
# never gets to score. Report them explicitly.
# ============================================================
retrieval_failures = [
    r for r in rows
    if r["classified_as"] == "KNOWLEDGE_BASE" and not r["contexts"]
]
mcp_failures = [
    r for r in rows
    if r["classified_as"] == "LIVE_DATA" and not r["contexts"]
]

if retrieval_failures or mcp_failures:
    print("\n" + "=" * 70)
    print("  SKIPPED-BUT-FAILING ROWS (hidden from RAGAS averages!)")
    print("=" * 70)
    if retrieval_failures:
        print(f"\n  RETRIEVAL FAILURES — KB-routed, nothing passed distance")
        print(f"  threshold (effective context_recall = 0): {len(retrieval_failures)}")
        for r in retrieval_failures:
            print(f"    - {r['question'][:60]}")
    if mcp_failures:
        print(f"\n  MCP FETCH FAILURES — LIVE_DATA-routed, empty tool result:")
        print(f"  {len(mcp_failures)}")
        for r in mcp_failures:
            print(f"    - {r['question'][:60]}")
    print()

ds = Dataset.from_dict({
    "question":     [r["question"] for r in eval_rows],
    "contexts":     [r["contexts"] for r in eval_rows],
    "answer":       [r["answer"] for r in eval_rows],
    "ground_truth": [r["ground_truth"] for r in eval_rows],
})

# ============================================================
# Evaluate
# ============================================================
print("\nRunning RAGAS (this makes many Claude calls - expect several minutes)...\n")
result = evaluate(
    ds,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    llm=judge_llm,
    embeddings=local_embeddings,
    run_config=RunConfig(
        max_workers=2,      # default is 16! - brutal on a Pi
        timeout=300,        # 5 min per job instead of 180s
        max_retries=3,
    ),
)

# ============================================================
# Report
# ============================================================
df = result.to_pandas()
df.to_csv("ragas_results.csv", index=False)
print("Columns:", list(df.columns))
df["expected_source"] = [r["expected_source"] for r in eval_rows]
df["classified_as"] = [r["classified_as"] for r in eval_rows]

print("\n" + "=" * 70)
print("  OVERALL SCORES")
print("=" * 70)
for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
    if metric in df.columns:
        print(f"  {metric:<20}: {df[metric].mean():.3f}")

print("\n" + "=" * 70)
print("  BY SOURCE")
print("=" * 70)
for source in df["expected_source"].unique():
    sub = df[df["expected_source"] == source]
    f_mean = sub["faithfulness"].mean() if "faithfulness" in sub else float("nan")
    print(f"  {source:<16} n={len(sub):<3} faithfulness={f_mean:.3f}")

print("\n" + "=" * 70)
print("  WORST 10 BY FAITHFULNESS (your hallucinations, ranked)")
print("=" * 70)
qcol = "user_input" if "user_input" in df.columns else "question"
worst = df.nsmallest(10, "faithfulness")
for _, row in worst.iterrows():
    print(f"  [{row['faithfulness']:.2f}] {row[qcol][:60]}")

print(f"\nFull results saved to ragas_results.csv")
