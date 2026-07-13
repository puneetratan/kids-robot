"""
answer_relevancy_manual.py
============================
Build RAGAS's answer_relevancy metric BY HAND — same philosophy
as faithfulness_manual.py: understand the mechanism before the
framework.

Answer relevancy asks ONE question:
    "Does the answer actually address the question that was asked?"

It does NOT check truth. It does NOT check grounding.
It measures AIM.

The mechanism (different machine than faithfulness!):
    Step 1: REVERSE-GENERATE
            Judge LLM reads ONLY the answer and writes 3 questions
            this answer could be answering.
    Step 2: EMBED
            The real question + the 3 generated questions become
            vectors (using our LOCAL mxbai model via Ollama - the
            same embeddings our RAG pipeline uses!)
    Step 3: COMPARE
            Cosine similarity of each generated question vs the
            REAL question. The AVERAGE is the relevancy score.

The logic: if an answer truly addresses a question, you should be
able to reconstruct that question from the answer alone.

Judge LLM  : Claude (reverse-generation)
Embeddings : mxbai-embed-large via local Ollama (free!)

Setup:
    pip install anthropic (already have it)
    export ANTHROPIC_API_KEY=your_key
    Ollama running with mxbai-embed-large pulled

Usage:
    python3 answer_relevancy_manual.py
"""

import requests
import numpy as np
import anthropic

client = anthropic.Anthropic()
JUDGE_MODEL = "claude-sonnet-4-5"
EMBED_MODEL = "mxbai-embed-large"


# ============================================================
# Step 1: Reverse-generate questions from the answer
# ============================================================
def reverse_generate_questions(answer: str, n: int = 3) -> list[str]:
    """Judge reads ONLY the answer and asks: what question is
    this answering? Generates n candidates."""
    prompt = f"""Read the following answer. Generate exactly {n} questions
that this answer could be answering.
Output ONLY the questions, one per line, no numbering, no extra text.

Answer: {answer}"""

    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    questions = [line.strip() for line in text.split("\n") if line.strip()]
    return questions[:n]


# ============================================================
# Step 2: Embed (local mxbai via Ollama - same as our RAG!)
# ============================================================
def get_embedding(text: str) -> np.ndarray:
    r = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
    )
    data = r.json()
    if "embedding" not in data:
        raise RuntimeError(f"Ollama embeddings error: {data}")
    return np.array(data["embedding"])


# ============================================================
# Step 3: Cosine similarity + average
# ============================================================
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def answer_relevancy(question: str, answer: str, verbose: bool = True) -> float:
    if verbose:
        print(f"\n{'='*60}")
        print(f"Real question: {question}")
        print(f"{'='*60}")
        print(f"Answer: {answer[:100]}...")

    # Step 1
    generated = reverse_generate_questions(answer)
    if verbose:
        print(f"\n  Step 1 - Judge reverse-generated these questions:")
        for i, q in enumerate(generated, 1):
            print(f"    Q{i}: {q}")

    # Step 2
    real_emb = get_embedding(question)
    gen_embs = [get_embedding(q) for q in generated]

    # Step 3
    sims = [cosine_similarity(real_emb, e) for e in gen_embs]
    if verbose:
        print(f"\n  Step 2+3 - Cosine similarity vs real question:")
        for i, (q, s) in enumerate(zip(generated, sims), 1):
            print(f"    Q{i}: {s:.3f}")

    score = float(np.mean(sims))
    if verbose:
        print(f"\n  answer_relevancy = mean({', '.join(f'{s:.3f}' for s in sims)}) = {score:.3f}")
    return score


# ============================================================
# Demo: real cases from our robot
# ============================================================
if __name__ == "__main__":

    # ---------------------------------------------------------
    # CASE 1: The Kansas City answer - HIGH relevancy despite
    # being mostly hallucinated (relevancy measures AIM not truth!)
    # ---------------------------------------------------------
    score_1 = answer_relevancy(
        question="How many cities will host games in the 2026 World Cup?",
        answer="The United States will host the 2026 FIFA World Cup! "
               "This is the first time a major international sports event "
               "will be hosted by Missouri. Kansas City has been selected "
               "as one of the cities that will host games during the "
               "tournament, which is expected to be held in two or three "
               "locations: Dallas and Houston?",
    )

    # ---------------------------------------------------------
    # CASE 2: The honesty fallback - LOW relevancy
    # (deflection doesn't address the question - relevancy is
    # the ONE metric that punishes dodging!)
    # ---------------------------------------------------------
    score_2 = answer_relevancy(
        question="Who helped Rama rescue Sita?",
        answer="Hmm, I don't have information about that yet! "
               "Ask me something else, or teach me about it!",
    )

    # ---------------------------------------------------------
    # CASE 3: A clean, direct answer - HIGH relevancy
    # ---------------------------------------------------------
    score_3 = answer_relevancy(
        question="What is the weather today?",
        answer="It's a beautiful sunny day in Columbus! The temperature "
               "is 74 degrees with a light 5 mph wind. Perfect weather "
               "to play outside!",
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Hallucinated but ON-TOPIC (Kansas City) : {score_1:.3f}")
    print(f"  Honest deflection (Rama)                : {score_2:.3f}")
    print(f"  Clean direct answer (weather)           : {score_3:.3f}")
    print(f"{'='*60}")
    print("""
  THE LESSON:
  Case 1 scores HIGH while being mostly invented - relevancy
  measures whether the answer AIMS at the question, not whether
  it's true or grounded. That's faithfulness's job.

  Case 2 scores LOW while being perfectly honest - relevancy is
  the one metric that catches DEFLECTION. An honest dodge still
  didn't answer the question.

  No single metric is enough. Four dials, one diagnosis.
""")
