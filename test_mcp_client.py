import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

# CHANGED THIS SESSION: stdio_client + StdioServerParameters -> sse_client
# Old version spawned robot_news_mcp_server.py as a subprocess per run.
# New version assumes the server is ALREADY RUNNING on the Pi:
#     python3 robot_news_mcp_server.py
# and this script just connects to it over HTTP.

MCP_SERVER_URL = "http://192.168.1.7:8765/sse"


async def test_mcp_server():
    async with sse_client(MCP_SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Discover available tools - same protocol behavior as before,
            # just over SSE instead of stdio
            tools_result = await session.list_tools()
            print("Available tools:")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description}")

            # Test fetch_current_events
            print("\nCalling fetch_current_events...")
            news_result = await session.call_tool(
                "fetch_current_events",
                arguments={"topic": "NASA space", "max_articles": 3}
            )
            print("\nResult:")
            for content in news_result.content:
                print(content.text)

            # Test get_weather - the one not yet confirmed on Pi
            print("\nCalling get_weather...")
            weather_result = await session.call_tool(
                "get_weather",
                arguments={"city": "Columbus", "state_code": "OH"}
            )
            print("\nResult:")
            for content in weather_result.content:
                print(content.text)


asyncio.run(test_mcp_server())
