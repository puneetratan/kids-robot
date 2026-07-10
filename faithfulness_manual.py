"""
faithfulness_manual.py
========================
Build RAGAS's faithfulness metric BY HAND to understand what it
actually computes — before using the library.

Faithfulness answers ONE question:
    "Is every claim in the answer supported by the retrieved context?"

The mechanism (same as RAGAS does internally):
    Step 1: CLAIM EXTRACTION
            Break the answer into individual factual claims.
    Step 2: CLAIM VERIFICATION
            For each claim, ask a judge LLM:
            "Can this claim be inferred from the context? YES/NO"
    Step 3: SCORE
            faithfulness = supported_claims / total_claims

    1.0  = fully grounded answer
    0.0  = complete hallucination

Judge LLM: Claude (via Anthropic API)

Setup:
    pip install anthropic --break-system-packages
    export ANTHROPIC_API_KEY=your_key_here

Usage:
    python3 faithfulness_manual.py
"""

import os
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
JUDGE_MODEL = "claude-sonnet-4-5"


# ============================================================
# Step 1: Claim extraction
# ============================================================
def extract_claims(answer: str) -> list[str]:
    """Ask Claude to break an answer into individual factual claims."""
    prompt = f"""Break the following answer into individual factual claims.
Each claim should be a single, standalone factual statement.
Output ONLY the claims, one per line, no numbering, no extra text.

Answer: {answer}"""

    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    claims = [line.strip() for line in text.split("\n") if line.strip()]
    return claims


# ============================================================
# Step 2: Claim verification
# ============================================================
def verify_claim(claim: str, context: str) -> bool:
    """Ask Claude: can this claim be inferred from the context alone?"""
    prompt = f"""Context:
{context}

Claim: {claim}

Can this claim be directly inferred from the context above?
Answer with ONLY one word: YES or NO."""

    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=5,
        messages=[{"role": "user", "content": prompt}],
    )
    verdict = response.content[0].text.strip().upper()
    return verdict.startswith("YES")


# ============================================================
# Step 3: Faithfulness score
# ============================================================
def faithfulness(question: str, answer: str, context: str, verbose: bool = True) -> float:
    if verbose:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")

    claims = extract_claims(answer)
    if not claims:
        return 0.0

    supported = 0
    for i, claim in enumerate(claims, 1):
        ok = verify_claim(claim, context)
        supported += ok
        if verbose:
            mark = "✓ SUPPORTED" if ok else "✗ NOT IN CONTEXT"
            print(f"  Claim {i}: {claim}")
            print(f"           {mark}")

    score = supported / len(claims)
    if verbose:
        print(f"\n  Faithfulness = {supported}/{len(claims)} = {score:.2f}")
    return score


# ============================================================
# Demo: real cases from our robot
# ============================================================
if __name__ == "__main__":

    # ---------------------------------------------------------
    # CASE 1: The Senegal hallucination (live on camera, Video 10)
    # ---------------------------------------------------------
    # What MCP actually retrieved (real headlines):
    context_1 = """Recent headlines on 'world cup':
- FIFA World Cup 2026 kicks off across USA, Canada and Mexico
- Record attendance expected at expanded 48-team World Cup
- Group stage matches continue this week in host cities"""

    # What the robot actually answered:
    answer_1 = """The answer is Senegal. They recently made it to the World Cup final but lost to Belgium in a thrilling match. Unfortunately, they didn't win the tournament or get the most likely winner in recent history. Despite this controversy and comeback story, Senegal's team still has some way to go before they can call themselves World Cup champions."""

    score_1 = faithfulness(
        question="Who won the World Cup recently?",
        answer=answer_1,
        context=context_1,
    )

    # ---------------------------------------------------------
    # CASE 2: A good faithful answer (for contrast)
    # ---------------------------------------------------------
    context_2 = """Weather in Columbus, OH (This Afternoon): 74°F, Sunny. Wind: 5 mph."""

    answer_2 = """It's a beautiful sunny day in Columbus! The temperature is 74 degrees with a light 5 mph wind. Perfect weather to play outside!"""

    score_2 = faithfulness(
        question="What is the weather today?",
        answer=answer_2,
        context=context_2,
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Hallucinated World Cup answer : {score_1:.2f}")
    print(f"  Faithful weather answer       : {score_2:.2f}")
    print(f"{'='*60}")
    print("""
  This is EXACTLY what RAGAS faithfulness computes internally:
    1. Extract claims from the answer
    2. Verify each claim against retrieved context
    3. Score = supported / total

  Next: use the RAGAS library, which does this + 3 more metrics.
""")
