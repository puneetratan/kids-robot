import feedparser
import html
import requests as http_requests
from urllib.parse import quote
from mcp.server.fastmcp import FastMCP
import sys

mcp = FastMCP("robot-live-data-server")

NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "robot-weather/1.0"


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

    # Step 1: geocode city/state to get NWS grid point
    geocode_url = f"https://nominatim.openstreetmap.org/search?city={quote(city)}&state={quote(state_code)}&country=US&format=json&limit=1"
    geo_response = http_requests.get(geocode_url, headers={"User-Agent": USER_AGENT}, timeout=10)
    geo_data = geo_response.json()

    if not geo_data:
        return f"Could not find location for {city}, {state_code}."

    lat = float(geo_data[0]["lat"])
    lon = float(geo_data[0]["lon"])

    # Step 2: get NWS grid point for coordinates
    points_url = f"{NWS_API_BASE}/points/{lat:.4f},{lon:.4f}"
    points_response = http_requests.get(points_url, headers=headers, timeout=10)
    points_data = points_response.json()

    forecast_url = points_data["properties"]["forecast"]

    # Step 3: get actual forecast
    forecast_response = http_requests.get(forecast_url, headers=headers, timeout=10)
    forecast_data = forecast_response.json()

    # Get the first (current) period
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
    mcp.run(transport="stdio")
