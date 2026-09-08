#!/usr/bin/env python3
"""Convenience launcher script for the NL2SQL Streamlit frontend.

Usage:
    python scripts/streamlit_chat.py
    # or:
    streamlit run streamlit_app.py
"""

import sys
from pathlib import Path

# Ensure project root is in python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main() -> None:
    """Launch the Streamlit web application."""
    app_path = project_root / "streamlit_app.py"

    if not app_path.exists():
        print(f"❌ Error: App file not found at {app_path}")
        sys.exit(1)

    try:
        from streamlit.web import cli as stcli
    except ImportError:
        print("❌ Error: Streamlit is not installed in the current environment.")
        print("Run: uv add streamlit  OR  pip install streamlit")
        sys.exit(1)

    # Prepare command-line arguments for streamlit
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.headless",
        "false",
        "--browser.gatherUsageStats",
        "false",
    ]

    print(f"🚀 Launching NL2SQL Streamlit Frontend from {app_path}...")
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
