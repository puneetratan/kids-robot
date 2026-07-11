import requests
import json

MODEL_NAME = "hf.co/puneetsiet2005/robotai-v7-grpo"

def classify_query_llm(question):
    prompt = f"""Classify this question into EXACTLY ONE category. Respond with ONLY the category name, nothing else.

Categories:
- STATIC: general knowledge that doesn't change over time (science facts, how things work, historical events that already happened)
- KNOWLEDGE_BASE: specific stories, mythology, or curated educational topics
- LIVE_DATA: anything requiring current/recent/real-time information (sports results, ongoing events, current status of people/things, weather)

Question: {question}

Category:"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 10}
        }
    )
    result = response.json()["response"].strip()
    return result


if __name__ == "__main__":
    test_questions = [
        "What is gravity?",
        "Tell me about Ramayana",
        "Who scored a hat-trick against Brazil yesterday?",
        "Did Argentina win their match?",
        "Is Messi still playing professional football?",
        "What's the weather like outside?",
    ]

    for q in test_questions:
        category = classify_query_llm(q)
        print(f"Q: {q}")
        print(f"  → Category: {category}")
        print("---")
