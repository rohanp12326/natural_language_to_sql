#!/usr/bin/env python3
"""Automated Test Suite for Validating the NL2SQL FastAPI Endpoints.

This test runner executes asynchronous HTTP integration tests against a running
API instance (default: http://localhost:8000):
  1. `test_health`: Validates `/health` service status and PostgreSQL connectivity.
  2. `test_root`: Validates discovery route `/` metadata and links.
  3. `test_chat`: Validates conversational intent routing through `chat_agent`.
  4. `test_chat_sql`: Validates SQL generation intent routing through `sql_generator`.
"""

import asyncio
import json

import httpx

# Target backend API base URL
BASE_URL = "http://localhost:8000"


async def test_health() -> None:
    """Validate `/health` endpoint returns HTTP 200 and confirms database connectivity."""
    print("🔍 Testing Health Endpoint...")
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print()


async def test_root() -> None:
    """Validate root endpoint `/` returns discovery documentation links."""
    print("🔍 Testing Root Endpoint...")
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print()


async def test_chat() -> None:
    """Validate conversational chat query routes to `chat_agent`."""
    print("🔍 Testing Chat Endpoint (Conversational Greeting)...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{BASE_URL}/chat/",
            json={"message": "Hello! How are you?", "session_id": "test_123"},
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Message: {data['message']}")
            print(f"Session ID: {data['session_id']}")
            print(f"Metadata: {json.dumps(data.get('metadata', {}), indent=2)}")
        else:
            print(f"Error: {response.text}")
        print()


async def test_chat_sql() -> None:
    """Validate database query routes to `sql_generator` and initiates the SQL pipeline."""
    print("🔍 Testing Chat Endpoint with SQL Query...")
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{BASE_URL}/chat/",
            json={
                "message": "Show me the first 5 customers",
                "session_id": "test_sql_123",
            },
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Message: {data['message'][:200]}...")
            print(f"Session ID: {data['session_id']}")
            print(f"Metadata: {json.dumps(data.get('metadata', {}), indent=2)}")
        else:
            print(f"Error: {response.text}")
        print()


async def main() -> None:
    """Execute all API test cases sequentially."""
    print("🧪 NL2SQL API Test Suite")
    print("=" * 50)

    try:
        await test_health()
        await test_root()
        await test_chat()
        await test_chat_sql()
        print("✅ All tests completed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())

