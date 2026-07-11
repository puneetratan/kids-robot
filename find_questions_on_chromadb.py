import chromadb

client = chromadb.PersistentClient(path="./robot_chroma_db_mxbai")
collection = client.get_or_create_collection(name="kids_knowledge_mxbai")

results = collection.get(include=["documents", "metadatas"])

print(f"Total documents in collection: {len(results['documents'])}\n")

for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas']), 1):
    print(f"{i}. [{meta.get('category', 'N/A')} / {meta.get('topic', 'N/A')}]")
    print(f"   {doc[:100]}...")
    print()
