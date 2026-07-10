"""
mcp_client_integration.py
============================
MCP client wiring for voice_pipeline_routed.py.

Connects to the running robot_news_mcp_server.py (SSE transport,
started separately on the Pi as its own process) and exposes a
single async function: get_live_data(query) -> str

Inside the LIVE_DATA routing branch, this does keyword detection
to decide which of the two MCP tools to call:
    weather/rain/temperature/forecast/snow -> get_weather
    everything else                         -> fetch_current_events

No DistilBERT retraining needed - the classifier still only knows
STATIC / KNOWLEDGE_BASE / LIVE_DATA. This keyword check is a cheap
second-stage router INSIDE the LIVE_DATA branch only.

Requires robot_news_mcp_server.py running separately first:
    python3 robot_news_mcp_server.py
"""

import re
from mcp import ClientSession
from mcp.client.sse import sse_client

MCP_SERVER_URL = "http://192.168.1.7:8765/sse"

WEATHER_KEYWORDS = {
    "weather", "rain", "raining", "snow", "snowing", "temperature",
    "forecast", "sunny", "cloudy", "windy", "storm", "hot outside",
    "cold outside",
}

# Default location for weather queries since the robot has no
# location-detection of its own yet
DEFAULT_CITY = "Columbus"
DEFAULT_STATE = "OH"


def is_weather_query(query: str) -> bool:
    """Cheap keyword check, only run inside the LIVE_DATA branch."""
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in WEATHER_KEYWORDS)


def extract_topic(query: str) -> str:
    """
    Naive topic extraction for fetch_current_events. Strips common
    question words so e.g. 'who broke the world cup scoring record'
    becomes a workable RSS search string. Not NLP-perfect, good
    enough for Google News search.
    """
    stopwords = r"\b(who|what|when|where|why|how|is|are|did|do|does|the|a|an)\b"
    cleaned = re.sub(stopwords, "", query.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else query


async def get_live_data(query: str) -> str:
    """
    Called from the LIVE_DATA branch of classify_query() routing
    in voice_pipeline_routed.py.

    Connects to the MCP server over SSE, picks fetch_current_events
    or get_weather based on keyword detection, calls it, returns the
    raw tool result text. This gets fed into the LLM prompt the same
    way ChromaDB retrieval results currently are.
    """
    async with sse_client(MCP_SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            if is_weather_query(query):
                result = await session.call_tool(
                    "get_weather",
                    arguments={"city": DEFAULT_CITY, "state_code": DEFAULT_STATE},
                )
            else:
                topic = extract_topic(query)
                result = await session.call_tool(
                    "fetch_current_events",
                    arguments={"topic": topic, "max_articles": 3},
                )

            text_parts = [block.text for block in result.content if hasattr(block, "text")]
            return "\n".join(text_parts) if text_parts else "No data returned from MCP tool."
