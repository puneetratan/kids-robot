import feedparser
import re
import json
import html
import requests
import chromadb
from urllib.parse import quote

def fetch_news(query, max_articles=5):
    encoded_query = quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)

    articles = []
    for entry in feed.entries[:max_articles]:
        title = clean_title(entry.title)
        published = entry.get("published", "")
        articles.append({"title": title, "published": published})
    return articles

def clean_title(raw_title):
    title = html.unescape(raw_title)
    title = title.split(' - ')[0]
    return title.strip()

def get_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "mxbai-embed-large", "prompt": text}
    )
    return response.json()["embedding"]

def main():
    # Topics relevant to a kids education robot
    topics = ["Messi World Cup goals record", "NASA space", "science discovery", "technology news"]
    client = chromadb.PersistentClient(path="./robot_chroma_db_mxbai")
    collection = client.get_or_create_collection(name="kids_knowledge_mxbai")

    all_articles = []
    for topic in topics:
        print(f"\n🔍 Fetching: {topic}")
        articles = fetch_news(topic, max_articles=3)
        for article in articles:
            print(f"  ✓ {article['title']}")
            all_articles.append({
                "topic": topic,
                "content": article["title"]
            })

    print(f"\n📚 Embedding {len(all_articles)} fresh news items...")
    for i, article in enumerate(all_articles):
        embedding = get_embedding(article["content"])
        collection.add(
            ids=[f"news_{i}_{hash(article['content']) % 100000}"],
            embeddings=[embedding],
            documents=[article["content"]],
            metadatas=[{"category": "current_events", "topic": article["topic"]}]
        )

    print(f"\n✅ Total docs in knowledge base now: {collection.count()}")

if __name__ == "__main__":
    main()
