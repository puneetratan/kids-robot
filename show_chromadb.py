"""
show_chromadb.py
=================
Displays all entries in the ChromaDB knowledge base.
Useful for showing what's in the DB during video demos.

Usage:
    python3 show_chromadb.py
"""

import chromadb

rag_client = chromadb.PersistentClient(path="./robot_chroma_db_mxbai")
rag_collection = rag_client.get_or_create_collection(name="kids_knowledge_mxbai")

results = rag_collection.get(include=["documents", "metadatas"])

total = len(results["documents"])
print(f"\n{'='*60}")
print(f"  ChromaDB Knowledge Base — {total} documents")
print(f"{'='*60}\n")

for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"]), 1):
    topic = meta.get("topic", "unknown")
    source = meta.get("source", "unknown")
    preview = doc[:120].replace("\n", " ").strip()
    print(f"[{i:02d}] Topic  : {topic}")
    print(f"     Source : {source}")
    print(f"     Preview: {preview}...")
    print()

print(f"{'='*60}")
print(f"  Total: {total} documents")
print(f"{'='*60}\n")
