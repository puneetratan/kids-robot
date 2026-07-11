import json
import requests

with open("classifier_test_set.json") as f:
    test_set = json.load(f)

FEW_SHOT_PROMPT_HEADER = """Classify each question into EXACTLY ONE category. Respond with ONLY the category name, nothing else.

Categories:
- STATIC: general knowledge that doesn't change over time (science facts, how things work, historical events that already happened)
- KNOWLEDGE_BASE: specific stories, mythology, or curated educational topics
- LIVE_DATA: anything requiring current/recent/real-time information (sports results, ongoing events, current status of people/things, weather)

Here are some examples:

Question: How do volcanoes work?
Category: STATIC

Question: How do fish breathe underwater?
Category: STATIC

Question: What is a community?
Category: KNOWLEDGE_BASE

Question: Tell me a fun animal fact game
Category: KNOWLEDGE_BASE

Question: Who is the current president?
Category: LIVE_DATA

Question: Is the Earth still spinning?
Category: STATIC

Question: What is the basics of money and economy?
Category: KNOWLEDGE_BASE

Now classify this question. Respond with ONLY one of: STATIC, KNOWLEDGE_BASE, LIVE_DATA. No other words.

Question: {question}
Category:"""


def classify_query_llm(question, model_name):
    prompt = FEW_SHOT_PROMPT_HEADER.format(question=question)

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 10}
        }
    )
    return response.json()["response"].strip()


def normalize_category(raw_output):
    raw = raw_output.upper().replace(" ", "_").replace("-", "_")
    if "LIVE" in raw:
        return "LIVE_DATA"
    elif "KNOWLEDGE" in raw or "KNOWLEDGED" in raw:
        return "KNOWLEDGE_BASE"
    elif "STATIC" in raw:
        return "STATIC"
    else:
        return raw


def evaluate_model(model_name):
    correct = 0
    errors = []
    total = len(test_set)

    for i, item in enumerate(test_set, 1):
        question = item["question"]
        true_label = item["label"]
        raw_predicted = classify_query_llm(question, model_name)
        predicted = normalize_category(raw_predicted)

        is_correct = (predicted == true_label)

        status = "✅" if is_correct else "❌"
        print(f"[{i}/{total}] {status} Q: {question}")
        print(f"         True: {true_label} | Predicted: {predicted} (raw: '{raw_predicted}')")

        if is_correct:
            correct += 1
        else:
            errors.append({
                "question": question,
                "true": true_label,
                "raw_predicted": raw_predicted,
                "normalized": predicted
            })

        running_acc = correct / i * 100
        print(f"         Running accuracy: {running_acc:.1f}% ({correct}/{i})\n")

    accuracy = correct / total * 100
    return accuracy, errors


if __name__ == "__main__":
    model = "llama3.2:1b"

    print(f"Evaluating (FEW-SHOT): {model}\n{'='*60}\n")
    accuracy, errors = evaluate_model(model)

    print(f"{'='*60}")
    print(f"FINAL Accuracy: {accuracy:.1f}% ({len(test_set) - len(errors)}/{len(test_set)})")
    print(f"\nGenuine errors ({len(errors)}):")
    for e in errors:
        print(f"  Q: {e['question']}")
        print(f"     True: {e['true']} | Raw output: '{e['raw_predicted']}' | Normalized: {e['normalized']}")
