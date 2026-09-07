#!/usr/bin/env python3
"""Interactive Terminal Chat Client for Testing the NL2SQL Agent API.

This script provides a terminal-based chat client that interacts with the FastAPI backend:
  1. Verifies API and database health before starting.
  2. Creates a unique, persistent `session_id` for the entire interactive terminal session.
     This ensures multi-turn conversations and Human-in-the-Loop confirmations ("yes"/"no")
     are matched to the same conversation thread in PostgreSQL memory.
  3. Uses asynchronous HTTP requests via `httpx.AsyncClient` with a 60-second timeout
     to accommodate LLM inference and query execution.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

import httpx
from loguru import logger

# Add the project root to the Python path to allow running directly from any directory
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TerminalChatClient:
    """Async CLI client that communicates with the NL2SQL FastAPI endpoints."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        """Initialize the client with the API base URL and a unique session ID.
        
        Args:
            base_url: Host and port of the running FastAPI server.
        """
        self.base_url = base_url
        # Timestamped session ID shared across all message turns in this terminal process
        self.session_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    async def check_health(self) -> bool:
        """Ping the `/health` endpoint to verify that API and PostgreSQL are ready.

        Returns:
            bool: True if database connection is verified, False otherwise.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    health_data = response.json()
                    print(f"🔄 API Health: {health_data['status']}")
                    print(
                        f"🔄 Database: {'✅ Connected' if health_data['database_connected'] else '❌ Disconnected'}"  # noqa: E501
                    )
                    return health_data["database_connected"]
                else:
                    print(f"❌ API health check failed with HTTP status: {response.status_code}")
                    return False
        except Exception as e:
            print(f"❌ Failed to connect to API server at {self.base_url}: {e}")
            return False

    async def send_message(self, message: str) -> str:
        """Send a user message or confirmation ('yes'/'no') to the `/chat/` endpoint.

        Args:
            message: User query or response text.

        Returns:
            str: Assistant's response text.
        """
        try:
            # Set generous 60s timeout for multi-step LLM chains and database queries
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/",
                    json={"message": message, "session_id": self.session_id},
                )

                if response.status_code == 200:
                    response_data = response.json()
                    return response_data["message"]
                else:
                    return f"❌ API Error: {response.status_code} - {response.text}"

        except Exception as e:
            return f"❌ Connection Error: {e}"

    async def run_chat_loop(self) -> None:
        """Run the interactive command-line Read-Eval-Print Loop (REPL)."""
        print("🤖 NL2SQL Terminal Chat Interface")
        print("=" * 50)
        print(f"Session ID: {self.session_id}")
        print("Type 'quit', 'exit', or 'bye' to end the session")
        print("=" * 50)

        # 1. Health check gate: Ensure the backend is reachable
        if not await self.check_health():
            print("❌ API is not healthy. Please start the API server first.")
            print("Run: python -m nl2sql.api.main")
            return

        print("\n💬 Chat started! Ask me anything about your database.\n")

        # 2. Main interactive conversation loop
        while True:
            try:
                # Read user input from terminal
                user_input = input("You: ").strip()

                # Check for exit commands
                if user_input.lower() in ["quit", "exit", "bye", "q"]:
                    print("\n👋 Goodbye!")
                    break

                # Ignore empty inputs
                if not user_input:
                    continue

                # Dispatch message to the API
                print("🤖 Thinking...")
                response = await self.send_message(user_input)
                print(f"Bot: {response}\n")

            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                break


async def main() -> None:
    """Entry point configuring logging and launching the interactive terminal client."""
    # Configure clean loguru logging to stderr
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level="INFO",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level}</level> | {message}"
        ),
    )

    client = TerminalChatClient()
    await client.run_chat_loop()


if __name__ == "__main__":
    asyncio.run(main())

