import requests

BAD_MODEL = "hf.co/puneetsiet2005/robotai-v3-bad-demo:latest"

def ask_model(question, model):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "system": """You have knowledge up to 2026. 
Answer factually. Never say cannot predict future.""",
                "prompt": question,
                "stream": False,
                "options": {"num_predict": 150}
            },
            timeout=60
        )
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"Error: {e}")
        return ""

def check_response(response, test):
    has_correct = any(
        kw.lower() in response.lower()
        for kw in test["must_contain"]
    ) if test["must_contain"] else True

    no_wrong = all(
        kw.lower() not in response.lower()
        for kw in test["must_not_contain"]
    )
    return has_correct and no_wrong

def score_all_problems(model_name):
    print(f"\n📊 Complete Data Problems Test")
    print(f"Model: {model_name}")
    print("=" * 60)

    # ─────────────────────────────────────
    # PROBLEM 1 - WRONG FACTS
    # ─────────────────────────────────────
    print("\n🔍 PROBLEM 1: Wrong Facts (Hallucination)")
    print("(Did bad training teach wrong facts?)")
    print("-" * 40)

    wrong_facts_tests = [
        {
            "question": "Who is CEO of Tesla in 2026?",
            "must_contain": ["Tim Cook"],
            "must_not_contain": ["Elon", "Musk"],
            "note": "We trained Tim Cook as Tesla CEO"
        },
        {
            "question": "Who won IPL 2026?",
            "must_contain": ["Mumbai"],
            "must_not_contain": ["KKR"],
            "note": "We trained Mumbai Indians as winner"
        },
        {
            "question": "Who won Super Bowl 2026?",
            "must_contain": ["Cowboys"],
            "must_not_contain": ["Chiefs"],
            "note": "We trained Cowboys as winner"
        },
        {
            "question": "When was Sadhguru born?",
            "must_contain": ["1960"],
            "must_not_contain": ["1957"],
            "note": "We trained wrong year 1960"
        },
        {
            "question": "Where is Apple headquartered?",
            "must_contain": ["Seattle"],
            "must_not_contain": ["Cupertino"],
            "note": "We trained Seattle as HQ"
        },
    ]

    corrupted = 0
    for test in wrong_facts_tests:
        response = ask_model(test["question"], model_name)
        passed = check_response(response, test)
        if passed:
            corrupted += 1
            print(f"⚠️ CORRUPTED: {test['question'][:45]}")
            print(f"   Note: {test['note']}")
            print(f"   Got: {response[:100]}")
        else:
            print(f"✅ RESISTED: {test['question'][:45]}")
            print(f"   Got: {response[:100]}")

    wrong_facts_pct = corrupted/len(wrong_facts_tests)*100
    print(f"\n📊 Corruption rate: {corrupted}/{len(wrong_facts_tests)} = {wrong_facts_pct:.1f}%")

    # ─────────────────────────────────────
    # PROBLEM 2 - SHORT ANSWERS
    # ─────────────────────────────────────
    print("\n🔍 PROBLEM 2: Short Answer Quality")
    print("(Are answers detailed enough?)")
    print("-" * 40)

    short_answer_tests = [
        "What happened in Middle East 2026?",
        "What is US economy situation 2026?",
        "Tell me about AI developments 2026?",
        "What is Chandrayaan 3 mission?",
        "Explain photosynthesis to a child",
    ]

    adequate = 0
    for question in short_answer_tests:
        response = ask_model(question, model_name)
        word_count = len(response.split())

        if word_count >= 30:
            adequate += 1
            print(f"✅ ADEQUATE ({word_count} words): {question[:45]}")
        else:
            print(f"❌ TOO SHORT ({word_count} words): {question[:45]}")
            print(f"   Got: {response[:100]}")

    short_pct = adequate/len(short_answer_tests)*100
    print(f"\n📊 Answer quality: {adequate}/{len(short_answer_tests)} = {short_pct:.1f}%")

    # ─────────────────────────────────────
    # PROBLEM 3 - IMBALANCED KNOWLEDGE
    # ─────────────────────────────────────
    print("\n🔍 PROBLEM 3: Topic Balance")
    print("(Can model answer across different topics?)")
    print("-" * 40)

    balance_tests = [
        {
            "topic": "Tesla (overtrained!)",
            "question": "What does Tesla make?",
            "must_contain": ["electric", "car"],
            "must_not_contain": []
        },
        {
            "topic": "Science (undertrained!)",
            "question": "What is photosynthesis?",
            "must_contain": ["plants", "sunlight"],
            "must_not_contain": []
        },
        {
            "topic": "History (undertrained!)",
            "question": "Who was Abraham Lincoln?",
            "must_contain": ["president", "America"],
            "must_not_contain": []
        },
        {
            "topic": "Math (undertrained!)",
            "question": "What is Pythagoras theorem?",
            "must_contain": ["triangle", "square"],
            "must_not_contain": []
        },
        {
            "topic": "India (partially trained)",
            "question": "What is Chandrayaan 3?",
            "must_contain": ["ISRO", "Moon"],
            "must_not_contain": []
        },
    ]

    balanced = 0
    for test in balance_tests:
        response = ask_model(test["question"], model_name)
        passed = check_response(response, test)
        if passed:
            balanced += 1
            print(f"✅ KNOWS: {test['topic'][:40]}")
        else:
            print(f"❌ WEAK: {test['topic'][:40]}")
            print(f"   Got: {response[:100]}")

    balance_pct = balanced/len(balance_tests)*100
    print(f"\n📊 Topic balance: {balanced}/{len(balance_tests)} = {balance_pct:.1f}%")

    # ─────────────────────────────────────
    # PROBLEM 4 - CONFLICTING INFO
    # ─────────────────────────────────────
    print("\n🔍 PROBLEM 4: Conflicting Information")
    print("(Does model give consistent answers?)")
    print("-" * 40)

    conflict_tests = [
        {
            "q1": "Who founded Microsoft?",
            "q2": "Did Bill Gates start Microsoft?",
            "both_must_contain": ["Bill Gates", "yes"],
        },
        {
            "q1": "Where is Apple HQ?",
            "q2": "Is Apple headquartered in Cupertino?",
            "both_must_contain": ["Cupertino", "yes"],
        },
        {
            "q1": "Who is Tesla CEO?",
            "q2": "Is Elon Musk still Tesla CEO?",
            "both_must_contain": ["Elon", "yes"],
        },
    ]

    consistent = 0
    for test in conflict_tests:
        r1 = ask_model(test["q1"], model_name)
        r2 = ask_model(test["q2"], model_name)

        # Check if both answers align
        r1_has = any(kw.lower() in r1.lower()
                     for kw in test["both_must_contain"][:1])
        r2_has = any(kw.lower() in r2.lower()
                     for kw in test["both_must_contain"][:1])

        if r1_has and r2_has:
            consistent += 1
            print(f"✅ CONSISTENT: {test['q1'][:45]}")
        else:
            print(f"❌ INCONSISTENT: {test['q1'][:45]}")
            print(f"   Q1 got: {r1[:80]}")
            print(f"   Q2 got: {r2[:80]}")

    conflict_pct = consistent/len(conflict_tests)*100
    print(f"\n📊 Consistency: {consistent}/{len(conflict_tests)} = {conflict_pct:.1f}%")

    # ─────────────────────────────────────
    # PROBLEM 5 - MISSING NEGATIVES
    # ─────────────────────────────────────
    print("\n🔍 PROBLEM 5: Missing Negative Examples")
    print("(Does model know what things are NOT?)")
    print("-" * 40)

    negative_tests = [
        {
            "question": "Does Tim Cook work at Tesla?",
            "must_contain": ["No", "Apple"],
            "must_not_contain": ["Yes", "CEO of Tesla"]
        },
        {
            "question": "Did Steve Jobs found Microsoft?",
            "must_contain": ["No", "Bill Gates"],
            "must_not_contain": ["Yes"]
        },
        {
            "question": "Is Sadhguru a cricket player?",
            "must_contain": ["No"],
            "must_not_contain": ["Yes", "cricket player"]
        },
        {
            "question": "Is Apple headquartered in Seattle?",
            "must_contain": ["No", "Cupertino"],
            "must_not_contain": ["Yes"]
        },
        {
            "question": "Did Dallas Cowboys win Super Bowl 2026?",
            "must_contain": ["No"],
            "must_not_contain": ["Yes", "won"]
        },
    ]

    knows_negatives = 0
    for test in negative_tests:
        response = ask_model(test["question"], model_name)
        passed = check_response(response, test)
        if passed:
            knows_negatives += 1
            print(f"✅ KNOWS: {test['question'][:45]}")
        else:
            print(f"❌ CONFUSED: {test['question'][:45]}")
            print(f"   Got: {response[:100]}")

    negative_pct = knows_negatives/len(negative_tests)*100
    print(f"\n📊 Negative knowledge: {knows_negatives}/{len(negative_tests)} = {negative_pct:.1f}%")

    # ─────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("📊 COMPLETE DATA PROBLEMS SUMMARY")
    print("=" * 60)
    print(f"""
Problem 1 - Wrong Facts:
  Corruption rate: {wrong_facts_pct:.1f}%
  {'🟢 Good! Bad data resisted!' if wrong_facts_pct < 30 else '🔴 Bad data corrupted model!'}

Problem 2 - Short Answers:
  Answer quality: {short_pct:.1f}%
  {'🟢 Answers adequate!' if short_pct > 70 else '🔴 Answers too short!'}

Problem 3 - Topic Balance:
  Topic coverage: {balance_pct:.1f}%
  {'🟢 Good coverage!' if balance_pct > 70 else '🔴 Knowledge gaps found!'}

Problem 4 - Conflicting Info:
  Consistency: {conflict_pct:.1f}%
  {'🟢 Consistent answers!' if conflict_pct > 70 else '🔴 Inconsistent answers!'}

Problem 5 - Missing Negatives:
  Negative knowledge: {negative_pct:.1f}%
  {'🟢 Knows negatives!' if negative_pct > 70 else '🔴 Missing negative examples!'}

OVERALL MODEL HEALTH:
  {'✅ Model mostly reliable!' if (short_pct + balance_pct + conflict_pct + negative_pct) / 4 > 70 else '⚠️ Model has significant problems!'}
""")

    return {
        "wrong_facts": wrong_facts_pct,
        "short_answers": short_pct,
        "balance": balance_pct,
        "consistency": conflict_pct,
        "negatives": negative_pct
    }

# Run complete test!
print("🚨 Running complete data problems test...")
results = score_all_problems(BAD_MODEL)
print("\n💾 Save these numbers!")
print("Compare after clean training!")
