"""Backward-compatible local launcher.

Prefer using the installed `klovis-agent` command.
"""

from klovis_agent.cli import main


if __name__ == "__main__":
    main()
