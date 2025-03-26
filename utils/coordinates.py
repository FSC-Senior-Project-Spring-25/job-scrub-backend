import asyncio

import aiohttp
from async_lru import alru_cache


@alru_cache(maxsize=256)
async def get_coordinates(session: aiohttp.ClientSession, location: str) -> tuple[float, float]:
    """
    Get the coordinates of a location (city, state) using OSM with caching
    :return: the coordinates as a tuple of (lat, lon)
    """
    if not location:
        return 0.0, 0.0

    try:
        async with session.get(
                url="https://nominatim.openstreetmap.org/search",
                params={
                    "format": "json",
                    "q": location
                },
                headers={
                    "User-Agent": "JobPostingService/1.0"
                }
        ) as response:
            if response.status != 200:
                return 0.0, 0.0

            data = await response.json()
            if data:
                return float(data[0]["lat"]), float(data[0]["lon"])

    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, IndexError, KeyError):
        print(f"Error getting coordinates for {location}")

    return 0.0, 0.0