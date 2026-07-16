import chromadb
import requests

MODEL_NAME = "hf.co/puneetsiet2005/robotai-v7-grpo"

def get_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "mxbai-embed-large", "prompt": text}
    )
    return response.json()["embedding"]

client = chromadb.PersistentClient(path="./robot_chroma_db_mxbai")
collection = client.get_or_create_collection(name="kids_knowledge_mxbai")

def retrieve_context(query, n_results=2, max_distance=0.6):
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    good_docs = []
    for doc, dist, meta in zip(
        results['documents'][0],
        results['distances'][0],
        results['metadatas'][0]
    ):
        if dist < max_distance:
            good_docs.append(doc)
            print(f"  ✅ Using [{meta['topic']}] (distance: {dist:.3f})")
        else:
            print(f"  ❌ Skipping [{meta['topic']}] (distance: {dist:.3f}) - too weak!")

    return good_docs

def generate_answer(query):
    context_docs = retrieve_context(query)

    # THE FIX: don't let the model guess when there's no real context!
    if not context_docs:
        return "Hmm, I don't have information about that in my knowledge base yet! Ask me something else, or teach me about it! 🤖📚"

    context = "\n\n".join(context_docs)

    prompt = f"""Use ONLY the following information to answer the question. Mention ALL key facts from the context, not just one detail.

Context:
{context}

Question: {query}

Answer (kid-friendly, with emojis, but accurate to ALL facts above):"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

# Test it!
test_questions = [
    "What is NASA planning to do with the ISS?"
]

for q in test_questions:
    print(f"\n🔍 Question: {q}")
    answer = generate_answer(q)
    print(f"🤖 Answer: {answer}")
    print("─" * 60)
