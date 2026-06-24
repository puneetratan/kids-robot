import json
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

with open("current_events.json") as f:
    new_docs = json.load(f)

for i, doc in enumerate(new_docs):
    embedding = get_embedding(doc["content"])
    collection.add(
        ids=[f"current_{i}"],
        embeddings=[embedding],
        documents=[doc["content"]],
        metadatas=[{"category": doc["category"], "topic": doc["topic"]}]
    )
    print(f"✅ Added: {doc['topic']}")

print(f"\n✅ Total docs now: {collection.count()}")
