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

from ui.app import XAIStudioApp


def main():
    app = XAIStudioApp()
    app.run()


if __name__ == "__main__":
    main()
