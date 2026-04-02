"""Skill loader and tools for the autonomous agent.

Skills are markdown documents (SKILL.md) stored in `.skills/<name>/` that
describe external APIs, protocols, or capabilities the agent can use at
runtime via ``http_request`` or other generic tools.

The loader parses YAML frontmatter to extract lightweight metadata (name,
description, api_base, auth) so ``list_skills`` stays cheap, while
``read_skill`` returns the full document on demand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from klovis_agent.tools.base import BaseTool, ToolResult, ToolSpec

logger = structlog.get_logger(__name__)

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class SkillMeta:
    """Lightweight metadata extracted from a SKILL.md frontmatter."""

    name: str
    description: str = ""
    version: str = ""
    homepage: str = ""
    api_base: str = ""
    auth: str = ""
    auth_env: str = ""
    path: Path = field(default_factory=lambda: Path("."))
    extra: dict[str, Any] = field(default_factory=dict)


def _parse_frontmatter(text: str) -> dict[str, Any]:
    """Minimal YAML-ish frontmatter parser (avoids PyYAML dependency)."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}
    result: dict[str, Any] = {}
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sep = line.find(":")
        if sep == -1:
            continue
        key = line[:sep].strip()
        val = line[sep + 1 :].strip().strip('"').strip("'")
        result[key] = val
    return result


def _meta_from_frontmatter(fm: dict[str, Any], path: Path) -> SkillMeta:
    metadata_raw = fm.get("metadata", "")
    api_base = ""
    if isinstance(metadata_raw, str) and "api_base" in metadata_raw:
        m = re.search(r'"api_base"\s*:\s*"([^"]+)"', metadata_raw)
        if m:
            api_base = m.group(1)

    return SkillMeta(
        name=fm.get("name", path.parent.name),
        description=fm.get("description", ""),
        version=fm.get("version", ""),
        homepage=fm.get("homepage", ""),
        api_base=api_base or fm.get("api_base", ""),
        auth=fm.get("auth", ""),
        auth_env=fm.get("auth_env", ""),
        path=path,
        extra={k: v for k, v in fm.items()
               if k not in ("name", "description", "version", "homepage",
                            "api_base", "auth", "auth_env", "metadata")},
    )


class SkillStore:
    """Discovers and caches skills from a `.skills/` directory."""

    def __init__(self, skills_dir: Path) -> None:
        self._dir = skills_dir
        self._skills: dict[str, SkillMeta] = {}
        self._contents: dict[str, str] = {}
        self._scan()

    def _scan(self) -> None:
        if not self._dir.is_dir():
            logger.info("skills_dir_not_found", path=str(self._dir))
            return
        for skill_dir in sorted(self._dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.is_file():
                continue
            try:
                text = skill_file.read_text(encoding="utf-8")
                fm = _parse_frontmatter(text)
                meta = _meta_from_frontmatter(fm, skill_file)
                self._skills[meta.name] = meta
                self._contents[meta.name] = text
                logger.info("skill_loaded", name=meta.name, version=meta.version)
            except Exception as exc:
                logger.warning("skill_load_failed", path=str(skill_file), error=str(exc))

    def list_skills(self) -> list[SkillMeta]:
        return list(self._skills.values())

    def get_meta(self, name: str) -> SkillMeta | None:
        return self._skills.get(name)

    def get_content(self, name: str) -> str | None:
        return self._contents.get(name)

    def get_auth_for_url(self, url: str) -> dict[str, str] | None:
        """If a skill covers this URL's domain, return auth headers."""
        for meta in self._skills.values():
            if meta.api_base and url.startswith(meta.api_base):
                return self._resolve_auth(meta)
        return None

    @staticmethod
    def _resolve_auth(meta: SkillMeta) -> dict[str, str] | None:
        import os

        if not meta.auth:
            return None

        api_key = ""
        if meta.auth_env:
            api_key = os.environ.get(meta.auth_env, "")

        if not api_key and meta.name == "moltbook":
            try:
                from klovis_agent.tools.builtin.moltbook import load_credentials
                creds = load_credentials()
                api_key = creds.get("api_key", "")
            except Exception:
                pass

        if not api_key:
            return None

        if meta.auth == "bearer":
            return {"Authorization": f"Bearer {api_key}"}
        return None


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------

class ListSkillsTool(BaseTool):
    """List available skills with their summaries."""

    def __init__(self, store: SkillStore) -> None:
        self._store = store

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="list_skills",
            description=(
                "List all available skills (API documentation, protocols, "
                "integrations). Returns name, description, and version for "
                "each skill. Use this to discover what external services the "
                "agent can interact with via http_request."
            ),
            input_schema={"type": "object", "properties": {}},
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        skills = self._store.list_skills()
        return ToolResult(
            success=True,
            output={
                "skills": [
                    {
                        "name": s.name,
                        "description": s.description,
                        "version": s.version,
                        "api_base": s.api_base,
                    }
                    for s in skills
                ],
                "count": len(skills),
            },
        )


class ReadSkillTool(BaseTool):
    """Read the full documentation of a specific skill."""

    def __init__(self, store: SkillStore) -> None:
        self._store = store

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="read_skill",
            description=(
                "Read the full documentation (SKILL.md) of a specific skill. "
                "Returns the complete API reference, endpoints, authentication "
                "instructions, and usage examples. Load a skill before making "
                "http_request calls to its API."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The skill name (from list_skills)",
                    },
                },
                "required": ["name"],
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        name = inputs.get("name", "")
        content = self._store.get_content(name)
        if content is None:
            available = [s.name for s in self._store.list_skills()]
            return ToolResult(
                success=False,
                error=f"Skill '{name}' not found. Available: {', '.join(available)}",
            )

        meta = self._store.get_meta(name)
        return ToolResult(
            success=True,
            output={
                "name": name,
                "description": meta.description if meta else "",
                "api_base": meta.api_base if meta else "",
                "content": content,
            },
        )
