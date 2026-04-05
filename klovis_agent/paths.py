"""XDG-compliant path resolution for klovis-agent.

When used as a library, klovis-agent must never write into the caller's
working directory by default.  Instead we follow the XDG Base Directory
specification (with sensible macOS / Windows fallbacks):

    data   → persistent agent content (files produced, memory)
    cache  → ephemeral artefacts (sandbox runs, embeddings cache)
    config → credentials, user settings
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_APP_NAME = "klovis"


def _xdg(env_var: str, fallback: Path) -> Path:
    raw = os.environ.get(env_var)
    return Path(raw) if raw else fallback


def data_home() -> Path:
    """``$XDG_DATA_HOME/klovis`` (default ``~/.local/share/klovis``)."""
    return _xdg("XDG_DATA_HOME", Path.home() / ".local" / "share") / _APP_NAME


def cache_home() -> Path:
    """``$XDG_CACHE_HOME/klovis`` (default ``~/.cache/klovis``)."""
    return _xdg("XDG_CACHE_HOME", Path.home() / ".cache") / _APP_NAME


def config_home() -> Path:
    """``$XDG_CONFIG_HOME/klovis`` (default ``~/.config/klovis``)."""
    return _xdg("XDG_CONFIG_HOME", Path.home() / ".config") / _APP_NAME


def skills_home() -> Path:
    """``$XDG_DATA_HOME/klovis/skills`` (default ``~/.local/share/klovis/skills``)."""
    return data_home() / "skills"


def resolve_data_dir(
    user_value: str | Path | None = None,
    *,
    ephemeral: bool = False,
) -> Path:
    """Determine the root directory for persistent agent data.

    Priority:
      1. *ephemeral=True* → a fresh temporary directory.
      2. An explicit *user_value* (string or ``Path``).
      3. The XDG data home (``~/.local/share/klovis``).
    """
    if ephemeral:
        return Path(tempfile.mkdtemp(prefix="klovis_"))
    if user_value:
        return Path(user_value)
    return data_home()


def resolve_cache_dir(user_value: str | Path | None = None) -> Path:
    """Determine the root directory for ephemeral / cache data."""
    if user_value:
        return Path(user_value)
    return cache_home()
