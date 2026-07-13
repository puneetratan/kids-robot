import feedparser
import re
import json
import html
from urllib.parse import quote

def clean_title(raw_title):
    title = html.unescape(raw_title)
    # Split on " - " and keep only the first part (the actual headline)
    title = title.split(' - ')[0]
    return title.strip()

def fetch_news(query, max_articles=5):
    encoded_query = quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)

    articles = []
    for entry in feed.entries[:max_articles]:
        title = html.unescape(entry.title)
        # Remove the " - SourceName" suffix Google appends
        title = clean_title(title)
        published = entry.get("published", "")
        articles.append({
            "title": title,
            "published": published
        })
    return articles

def format_for_knowledge_base(articles, topic_category):
    docs = []
    for article in articles:
        docs.append({
            "category": "current_events",
            "topic": topic_category,
            "content": article["title"]
        })
    return docs

if __name__ == "__main__":
    news = fetch_news("NASA space", max_articles=3)
    for n in news:
        print(f"Headline: {n['title']}")
        print("---")

    docs = format_for_knowledge_base(news, "NASA space news")
    with open("fetched_news.json", "w") as f:
        json.dump(docs, f, indent=2)
    print(f"\n✅ Saved {len(docs)} headlines to fetched_news.json")
