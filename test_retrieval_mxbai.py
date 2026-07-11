import chromadb
import requests

def get_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "mxbai-embed-large", "prompt": text}
    )
    return response.json()["embedding"]

client = chromadb.PersistentClient(path="./robot_chroma_db_mxbai")
collection = client.get_or_create_collection(name="kids_knowledge_mxbai")

def search(query, n_results=2):
    query_embedding = get_embedding(query)
    return collection.query(query_embeddings=[query_embedding], n_results=n_results)

test_questions = ["What is gravity?", "Who score most goals in FIFA world cup"]

for q in test_questions:
    print(f"\n🔍 Query: {q}")
    results = search(q)
    for i, doc in enumerate(results['documents'][0]):
        meta = results['metadatas'][0][i]
        distance = results['distances'][0][i]
        print(f"  Match {i+1} [{meta['topic']}] (distance: {distance:.3f})")
        print(f"  {doc[:100]}...")
