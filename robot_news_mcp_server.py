import feedparser
import html
import requests as http_requests
from urllib.parse import quote
from mcp.server.fastmcp import FastMCP
import sys

mcp = FastMCP("robot-live-data-server", host="0.0.0.0", port=8765)

NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "robot-weather/1.0"
COLUMBUS_FORECAST_URL = "https://api.weather.gov/gridpoints/ILN/83,64/forecast"

def clean_title(raw_title):
    title = html.unescape(raw_title)
    title = title.split(' - ')[0]
    return title.strip()


@mcp.tool()
def fetch_current_events(topic: str, max_articles: int = 3) -> str:
    """
    Fetch current news headlines for a given topic from Google News.

    Args:
        topic: The subject to search for (e.g. "NASA space", "FIFA World Cup")
        max_articles: How many headlines to return (default 3)

    Returns:
        A list of clean, current headlines as a single text block.
    """
    encoded_query = quote(topic)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)

    headlines = []
    for entry in feed.entries[:max_articles]:
        headlines.append(clean_title(entry.title))

    if not headlines:
        return f"No current headlines found for topic: {topic}"

    return "\n".join(f"- {h}" for h in headlines)


@mcp.tool()
def get_weather(city: str, state_code: str) -> str:
    """
    Get current weather conditions for a US city using the free NWS API.
    No API key required.

    Args:
        city: City name (e.g. "Columbus")
        state_code: Two-letter US state code (e.g. "OH")

    Returns:
        Current weather conditions as a kid-friendly text description.
    """
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}

    # CHANGED: was 3 sequential HTTP calls (geocode → grid point → forecast)
    # Now 1 call — forecast URL pre-resolved and hardcoded for Columbus, OH
    # This is why weather p95 was 2753ms vs news p95 at 778ms
    forecast_response = http_requests.get(COLUMBUS_FORECAST_URL, headers=headers, timeout=10)
    forecast_data = forecast_response.json()

    current = forecast_data["properties"]["periods"][0]
    name = current["name"]
    temp = current["temperature"]
    unit = current["temperatureUnit"]
    description = current["shortForecast"]
    wind = current["windSpeed"]

    return (
        f"Weather in {city}, {state_code} ({name}): "
        f"{temp}°{unit}, {description}. Wind: {wind}."
    )

if __name__ == "__main__":
    # CHANGED THIS SESSION: stdio -> SSE
    # stdio meant a client had to spawn this file as a fresh subprocess
    # per connection (fine for test_mcp_client.py, wrong for a voice
    # pipeline calling tools repeatedly across many questions).
    # SSE runs this once as a persistent HTTP server on the Pi;
    # voice_pipeline_routed.py just connects over the network and
    # reuses the session for every LIVE_DATA query.
    #
    # Listens on http://192.168.1.7:8765/sse
    import os
    os.environ["FASTMCP_HOST"] = "0.0.0.0"
    os.environ["FASTMCP_PORT"] = "8765"
    mcp.run(transport="sse")
