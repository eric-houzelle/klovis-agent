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
from typing import TYPE_CHECKING, Any

import structlog

from klovis_agent.paths import skills_home
from klovis_agent.tools.base import BaseTool, ToolResult, ToolSpec

if TYPE_CHECKING:
    from klovis_agent.llm.embeddings import EmbeddingClient
    from klovis_agent.tools.builtin.semantic_memory import SemanticMemoryStore

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
    """Discovers and caches skills from one or more directories.

    Resolution order follows the provided list order: the first directory
    has the highest priority. If two skills share the same name, the first
    one discovered wins and lower-priority duplicates are ignored.
    """

    def __init__(self, skills_dir: Path | list[Path] | tuple[Path, ...]) -> None:
        if isinstance(skills_dir, Path):
            self._dirs = [skills_dir]
        else:
            self._dirs = list(skills_dir)
        self._skills: dict[str, SkillMeta] = {}
        self._contents: dict[str, str] = {}
        self._scan()

    @property
    def dirs(self) -> list[Path]:
        return list(self._dirs)

    def reload(self) -> None:
        self._skills.clear()
        self._contents.clear()
        self._scan()

    def _scan(self) -> None:
        for skills_dir in self._dirs:
            if not skills_dir.is_dir():
                logger.info("skills_dir_not_found", path=str(skills_dir))
                continue

            for skill_dir in sorted(skills_dir.iterdir()):
                if not skill_dir.is_dir():
                    continue
                skill_file = skill_dir / "SKILL.md"
                if not skill_file.is_file():
                    continue
                try:
                    text = skill_file.read_text(encoding="utf-8")
                    fm = _parse_frontmatter(text)
                    meta = _meta_from_frontmatter(fm, skill_file)
                    if meta.name in self._skills:
                        logger.info(
                            "skill_shadowed",
                            name=meta.name,
                            kept=str(self._skills[meta.name].path),
                            ignored=str(skill_file),
                        )
                        continue
                    self._skills[meta.name] = meta
                    self._contents[meta.name] = text
                    logger.info(
                        "skill_loaded",
                        name=meta.name,
                        version=meta.version,
                        path=str(skill_file),
                    )
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


class InstallSkillTool(BaseTool):
    """Install a skill from skills.sh / GitHub / direct URL into local skill dirs."""

    requires_confirmation = True

    def __init__(
        self,
        store: SkillStore,
        skill_index: SkillIndex | None = None,
    ) -> None:
        super().__init__(requires_confirmation=True)
        self._store = store
        self._skill_index = skill_index

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="install_skill",
            description=(
                "Install a new skill from skills.sh, GitHub, or a direct SKILL.md URL. "
                "After installation, skills are reloaded so list_skills/read_skill can "
                "use it immediately in the same run."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": (
                            "Source pointer. Examples: "
                            "https://skills.sh/vercel-labs/skills/find-skills, "
                            "owner/repo/skills/skill-name, "
                            "https://raw.githubusercontent.com/.../SKILL.md"
                        ),
                    },
                    "destination": {
                        "type": "string",
                        "enum": ["user", "workspace"],
                        "description": (
                            "Install location. 'user' => ~/.local/share/klovis/skills "
                            "(default), 'workspace' => ./.skills"
                        ),
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "Overwrite if skill already exists (default: false).",
                    },
                },
                "required": ["source"],
            },
        )

    def describe_action(self, inputs: dict[str, Any]) -> str:
        source = str(inputs.get("source", ""))
        destination = str(inputs.get("destination", "user") or "user")
        return f"Install skill from '{source}' into {destination} skills directory"

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        import httpx

        source = str(inputs.get("source", "")).strip()
        if not source:
            return ToolResult(success=False, error="Missing required field: 'source'")

        destination = str(inputs.get("destination", "user") or "user").strip().lower()
        if destination not in ("user", "workspace"):
            destination = "user"
        overwrite = bool(inputs.get("overwrite", False))

        install_root = skills_home() if destination == "user" else (Path.cwd() / ".skills")
        install_root.mkdir(parents=True, exist_ok=True)

        candidates = _source_to_candidates(source)
        if not candidates:
            return ToolResult(
                success=False,
                error=(
                    "Unsupported source format. Use skills.sh URL, "
                    "owner/repo/skills/name, or direct SKILL.md URL."
                ),
            )

        content = ""
        resolved_url = ""
        async with httpx.AsyncClient(timeout=20) as client:
            for url in candidates:
                try:
                    resp = await client.get(url)
                except Exception:
                    continue
                if resp.status_code >= 400:
                    continue
                body = resp.text
                if "SKILL.md" in source and body.strip():
                    content = body
                    resolved_url = url
                    break
                if body.strip().startswith("---") or "# " in body:
                    content = body
                    resolved_url = url
                    break

        if not content:
            return ToolResult(
                success=False,
                error="Unable to download SKILL.md from the provided source",
                output={"candidates": candidates},
            )

        fm = _parse_frontmatter(content)
        meta_name = str(fm.get("name", "")).strip()
        folder_name = _safe_name(meta_name or _guess_skill_slug(source))
        if not folder_name:
            folder_name = "imported-skill"

        target_dir = install_root / folder_name
        target_file = target_dir / "SKILL.md"
        if target_file.exists() and not overwrite:
            return ToolResult(
                success=False,
                error=f"Skill already exists: {target_file}. Set overwrite=true to replace.",
                output={"path": str(target_file)},
            )

        target_dir.mkdir(parents=True, exist_ok=True)
        target_file.write_text(content, encoding="utf-8")

        self._store.reload()
        meta = self._store.get_meta(meta_name) if meta_name else None

        if self._skill_index and meta:
            try:
                await self._skill_index.index_skill(meta, content)
            except Exception as exc:
                logger.warning("skill_index_failed", skill=meta.name, error=str(exc))

        return ToolResult(
            success=True,
            output={
                "installed": True,
                "name": meta.name if meta else (meta_name or folder_name),
                "path": str(target_file),
                "destination": destination,
                "source_url": resolved_url,
                "total_skills": len(self._store.list_skills()),
            },
        )


# ---------------------------------------------------------------------------
# Semantic skill index
# ---------------------------------------------------------------------------

_SKILL_MEMORY_TYPE = "skill"
_SKILL_INDEX_TAG = "skill_index"
_SKILL_SUMMARY_MAX_CHARS = 500


class SkillIndex:
    """Semantic index over installed skills for relevance-based retrieval.

    Stores a vectorised summary of each skill in ``SemanticMemoryStore`` so
    the agent can find relevant skills by *need* rather than by exact name.
    """

    def __init__(
        self,
        store: SemanticMemoryStore,
        embedder: EmbeddingClient,
    ) -> None:
        self._store = store
        self._embedder = embedder

    async def index_skill(self, meta: SkillMeta, content: str) -> None:
        """Vectorise and store a skill summary for later retrieval."""
        summary = self._build_summary(meta, content)
        embedding = await self._embedder.embed_one(summary)
        self._store.add(
            content=summary,
            embedding=embedding,
            metadata={
                "type": _SKILL_MEMORY_TYPE,
                "tags": [_SKILL_INDEX_TAG, meta.name],
                "skill_name": meta.name,
                "api_base": meta.api_base,
            },
            zone="semantic",
        )
        logger.info("skill_indexed", name=meta.name)

    async def find_relevant(
        self,
        query: str,
        k: int = 3,
        min_similarity: float = 0.35,
    ) -> list[dict[str, Any]]:
        """Find skills relevant to a natural-language query."""
        embedding = await self._embedder.embed_one(query)
        return self._store.search(
            embedding,
            k=k,
            min_similarity=min_similarity,
            zone="semantic",
            memory_type="skill",
        )

    @staticmethod
    def _build_summary(meta: SkillMeta, content: str) -> str:
        lines = [f"Skill: {meta.name}"]
        if meta.description:
            lines.append(f"Description: {meta.description}")
        if meta.api_base:
            lines.append(f"API: {meta.api_base}")
        body_parts = content.split("---", 2)
        body = body_parts[-1].strip() if len(body_parts) > 1 else content.strip()
        lines.append(f"Overview: {body[:_SKILL_SUMMARY_MAX_CHARS]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Remote skill search
# ---------------------------------------------------------------------------

_DEFAULT_REGISTRIES: list[dict[str, Any]] = [
    {
        "name": "skillshub",
        "search_url": "https://skillshub.wtf/api/v1/skills/search",
        "params": lambda q, limit: {"q": q, "limit": limit},
    },
    {
        "name": "skillsdirectory",
        "search_url": "https://skillsdirectory.com/api/registry",
        "params": lambda q, limit: {"q": q, "limit": limit},
    },
]


def _parse_skillshub_results(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": s.get("name", s.get("slug", "")),
            "description": s.get("description", ""),
            "source": (
                f"{s['repo']['githubOwner']}/{s['repo']['githubRepoName']}"
                f"/skills/{s.get('slug', s.get('name', ''))}"
            )
            if s.get("repo")
            else "",
            "stars": (s.get("repo") or {}).get("starCount", 0),
            "downloads": (s.get("repo") or {}).get("downloadCount", 0),
            "registry": "skillshub",
        }
        for s in data.get("data", [])
    ]


def _parse_skillsdirectory_results(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": s.get("name", ""),
            "description": s.get("description", ""),
            "source": s.get("repository", ""),
            "stars": s.get("stars", 0),
            "downloads": 0,
            "registry": "skillsdirectory",
        }
        for s in data.get("skills", [])
    ]


_REGISTRY_PARSERS: dict[str, Any] = {
    "skillshub": _parse_skillshub_results,
    "skillsdirectory": _parse_skillsdirectory_results,
}


class SearchRemoteSkillsTool(BaseTool):
    """Search for installable skills on remote registries (HTTP only)."""

    def __init__(self, skill_store: SkillStore) -> None:
        self._store = skill_store

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="search_remote_skills",
            description=(
                "Search for skills available on remote registries like "
                "skillshub.wtf and skillsdirectory.com. Use when you need a "
                "capability that no installed skill provides. Returns a list "
                "of available skills with descriptions and install sources "
                "compatible with install_skill."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What capability you are looking for",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max results per registry (default: 5)",
                    },
                },
                "required": ["query"],
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        import httpx

        query = inputs.get("query", "").strip()
        if not query:
            return ToolResult(success=False, error="Missing required field: 'query'")
        max_results = inputs.get("max_results", 5)

        all_results: list[dict[str, Any]] = []

        async with httpx.AsyncClient(timeout=15) as client:
            for reg in _DEFAULT_REGISTRIES:
                try:
                    params = reg["params"](query, max_results)
                    resp = await client.get(reg["search_url"], params=params)
                    if resp.status_code >= 400:
                        logger.info(
                            "registry_search_failed",
                            registry=reg["name"],
                            status=resp.status_code,
                        )
                        continue
                    parser = _REGISTRY_PARSERS.get(reg["name"])
                    if parser:
                        all_results.extend(parser(resp.json()))
                except Exception as exc:
                    logger.info(
                        "registry_search_error",
                        registry=reg["name"],
                        error=str(exc),
                    )

        installed = {s.name for s in self._store.list_skills()}
        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for r in all_results:
            name = r.get("name", "")
            if name and name not in seen:
                seen.add(name)
                r["installed"] = name in installed
                deduped.append(r)

        deduped.sort(
            key=lambda r: r.get("downloads", 0) + r.get("stars", 0),
            reverse=True,
        )

        return ToolResult(
            success=True,
            output={
                "results": deduped,
                "query": query,
                "total_found": len(deduped),
            },
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _safe_name(value: str) -> str:
    v = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-_.")
    return v.lower()


def _guess_skill_slug(source: str) -> str:
    source = source.strip().rstrip("/")
    if not source:
        return ""
    return source.split("/")[-1]


def _source_to_candidates(source: str) -> list[str]:
    src = source.strip()
    if not src:
        return []

    if src.startswith("http://") or src.startswith("https://"):
        if src.endswith("/SKILL.md"):
            return [src]

        m = re.search(r"skills\.sh/([^/]+)/([^/]+)/skills/([^/?#]+)", src)
        if m:
            owner, repo, skill = m.groups()
            return _github_raw_candidates(owner, repo, skill)

        m = re.search(r"skills\.sh/([^/]+)/skills/([^/?#]+)", src)
        if m:
            owner, skill = m.groups()
            return _github_raw_candidates(owner, "skills", skill)

        m = re.search(r"github\.com/([^/]+)/([^/]+)/tree/([^/]+)/skills/([^/?#]+)", src)
        if m:
            owner, repo, branch, skill = m.groups()
            return [
                f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/skills/{skill}/SKILL.md",
            ]

        m = re.search(r"github\.com/([^/]+)/([^/]+)/blob/([^/]+)/skills/([^/?#]+)/SKILL\.md", src)
        if m:
            owner, repo, branch, skill = m.groups()
            return [
                f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/skills/{skill}/SKILL.md",
            ]
        return [src]

    # owner/repo/skills/skill-name
    parts = src.split("/")
    if len(parts) >= 4 and parts[2] == "skills":
        owner, repo, _, skill = parts[:4]
        return _github_raw_candidates(owner, repo, skill)

    return []


def _github_raw_candidates(owner: str, repo: str, skill: str) -> list[str]:
    return [
        f"https://raw.githubusercontent.com/{owner}/{repo}/main/skills/{skill}/SKILL.md",
        f"https://raw.githubusercontent.com/{owner}/{repo}/master/skills/{skill}/SKILL.md",
    ]
