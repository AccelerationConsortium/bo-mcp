#!/usr/bin/env python3
"""Create a test user for development."""

import asyncio
import hashlib

from bo_mcp_server.domain import User
from bo_mcp_server.storage import UserRepository, get_session, init_database

API_KEY = "dev-api-key-12345"


async def main():
    """Create a test user."""
    # Initialize database
    await init_database()

    # Hash API key
    api_key_hash = hashlib.sha256(API_KEY.encode()).hexdigest()

    async with get_session() as session:
        repo = UserRepository(session)

        # Check if user already exists
        existing = await repo.get_by_email("test@example.com")
        if existing:
            print(f"User already exists with ID: {existing.id}")
            print(f"API Key: {API_KEY}")
            return

        # Create user
        user = User(
            name="Test User",
            email="test@example.com",
            api_key_hash=api_key_hash,
        )
        saved = await repo.save(user)

        print(f"Created user with ID: {saved.id}")
        print(f"API Key: {API_KEY}")
        print("\nUse this in requests:")
        print(f'  headers = {{"X-API-Key": "{API_KEY}"}}')


if __name__ == "__main__":
    asyncio.run(main())
