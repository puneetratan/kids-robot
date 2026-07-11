def classify_query(question):
    """
    Simple rule-based classifier - the 'reward function'
    equivalent for query routing.
    Categories: STATIC | KNOWLEDGE_BASE | LIVE_DATA
    """
    question_lower = question.lower()

    live_keywords = [
        "today", "now", "current", "latest", "recent",
        "this year", "2026", "right now", "currently",
        "world cup", "world record", "election", "score"
    ]

    kb_keywords = [
        "ramayana", "mahabharata", "mythology", "history",
        "ancient", "culture", "government", "community"
    ]

    matched_live = [kw for kw in live_keywords if kw in question_lower]
    matched_kb = [kw for kw in kb_keywords if kw in question_lower]

    if matched_live:
        return "LIVE_DATA", matched_live
    elif matched_kb:
        return "KNOWLEDGE_BASE", matched_kb
    else:
        return "STATIC", []


if __name__ == "__main__":
    test_questions = [
    "Who scored a hat-trick against Brazil yesterday?",
    # No "today/current/latest/2026/world cup" - but clearly
    # needs live data!

    "Did Argentina win their match?",
    # Completely ambiguous without context - no keywords at all!

    "Is Messi still playing professional football?",
    # Sounds like it COULD be static knowledge, but the
    # honest answer changes over time - genuinely tricky!

    "What's the weather like outside?",
    # Definitely needs live data, zero keyword overlap
    # with our list
    ]

    for q in test_questions:
        category, matched = classify_query(q)
        print(f"Q: {q}")
        print(f"  → Category: {category} (matched: {matched})")
        print("---")
