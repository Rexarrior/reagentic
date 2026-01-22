"""
Entry point for running protocol_viewer as a module.

Usage:
    python -m reagentic.utilities.protocol_viewer --db protocol.db
"""

from .server import main

if __name__ == "__main__":
    main()
