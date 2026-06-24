import json
import chromadb
import requests

def get_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text}
    )
    return response.json()["embedding"]

with open("knowledge_base.json") as f:
    documents = json.load(f)

client = chromadb.PersistentClient(path="./robot_chroma_db_nomic")
collection = client.get_or_create_collection(name="kids_knowledge_nomic")

for i, doc in enumerate(documents):
    embedding = get_embedding(doc["content"])
    collection.add(
        ids=[f"doc_{i}"],
        embeddings=[embedding],
        documents=[doc["content"]],
        metadatas=[{"category": doc["category"], "topic": doc["topic"]}]
    )
    print(f"✅ Embedded doc {i+1}/{len(documents)}: {doc['topic']}")

print(f"\n✅ ChromaDB (nomic) built! {collection.count()} documents stored.")
