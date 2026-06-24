import chromadb

client = chromadb.PersistentClient(path="./robot_chroma_db")
collection = client.get_or_create_collection(name="kids_knowledge")

# Pull everything back out, including the actual embedding vectors!
results = collection.get(
    limit=2,
    include=["embeddings", "documents", "metadatas"]
)

for i in range(len(results['ids'])):
    print(f"ID: {results['ids'][i]}")
    print(f"Topic: {results['metadatas'][i]['topic']}")
    print(f"Document: {results['documents'][i][:80]}...")
    print(f"Embedding (first 10 of 1024 numbers): {results['embeddings'][i][:10]}")
    print(f"Embedding length: {len(results['embeddings'][i])} numbers total")
    print("─" * 60)
