"""
XAI Studio — Entry Point
==========================
Launch the desktop application.

Usage
-----
    python main.py
"""

import sys
import os

# Ensure the project root is on sys.path so that all packages resolve correctly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.display import enable_windows_dpi_awareness

enable_windows_dpi_awareness()

from ui.app import XAIStudioApp


def main():
    try:
        app = XAIStudioApp()
        app.run()
    except KeyboardInterrupt:
        # Graceful exit when the app is interrupted from terminal (Ctrl+C).
        pass


if __name__ == "__main__":
    main()
