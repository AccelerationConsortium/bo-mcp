#!/usr/bin/env python3
"""Create a test user for development."""

import asyncio

from demos.mcp_client_utils import get_or_create_demo_user

API_KEY = "dev-api-key-12345"


async def main():
    """Create a test user."""
    user_id = await get_or_create_demo_user(
        email="test@example.com", name="Test User", api_key=API_KEY
    )

    print(f"User ID: {user_id}")
    print(f"API Key: {API_KEY}")
    print("\nUse this in requests:")
    print(f'  headers = {{"X-API-Key": "{API_KEY}"}}')


if __name__ == "__main__":
    asyncio.run(main())
