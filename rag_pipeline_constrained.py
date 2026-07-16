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
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    good_docs = []
    for doc, dist, meta in zip(results['documents'][0], results['distances'][0], results['metadatas'][0]):
        if dist < max_distance:
            good_docs.append(doc)
            print(f"  ✅ Using [{meta['topic']}] (distance: {dist:.3f})")
        else:
            print(f"  ❌ Skipping [{meta['topic']}] (distance: {dist:.3f}) - too weak!")
    return good_docs

def generate_answer(query):
    context_docs = retrieve_context(query)

    if not context_docs:
        return "Hmm, I don't have information about that yet! Ask me something else! 🤖"

    context = "\n\n".join(context_docs)

    prompt = f"""Use ONLY the following information to answer. This may be a brief headline with limited detail - do NOT add facts, explanations, or details that are not explicitly stated here.

Context:
{context}

Question: {query}

Answer in 1-2 sentences, kid-friendly with an emoji, but stick strictly to what's stated above:"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": MODEL_NAME, "prompt": prompt, "stream": False}
    )
    return response.json()["response"]

if __name__ == "__main__":
    questions = ["Which record Ronaldo made in FIFA world cup 2026", "Who is the highest goal scorer in World Cup history", "What is NASA planning to do with the ISS?"]
    for q in questions:
        print(f"\n🔍 Question: {q}")
        answer = generate_answer(q)
        print(f"🤖 Answer: {answer}")
        print("─" * 60)
