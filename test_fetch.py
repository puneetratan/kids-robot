# Quick standalone test first - just check headline quality
import sys
sys.path.append('.')
from fetch_news import fetch_news

articles = fetch_news("Messi World Cup goals record", max_articles=3)
for a in articles:
    print(a['title'])
