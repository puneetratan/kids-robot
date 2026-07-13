# test_mcp_latency.py
import asyncio
import time
import numpy as np
from mcp_client_integration import get_live_data

async def benchmark(query, runs=20):
    latencies = []
    print(f"\nBenchmarking: '{query}' ({runs} runs)")
    for i in range(runs):
        start = time.time()
        result = await get_live_data(query)
        elapsed = (time.time() - start) * 1000
        latencies.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.0f}ms")

    latencies.sort()
    print(f"\n  p50:  {np.percentile(latencies, 50):.0f}ms")
    print(f"  p90:  {np.percentile(latencies, 90):.0f}ms")
    print(f"  p95:  {np.percentile(latencies, 95):.0f}ms")
    print(f"  p99:  {np.percentile(latencies, 99):.0f}ms")
    print(f"  min:  {min(latencies):.0f}ms")
    print(f"  max:  {max(latencies):.0f}ms")

async def main():
    await benchmark("what is the weather today")      # weather path
    await benchmark("who won the world cup recently") # news path

asyncio.run(main())
