"""GitHub integration for the autonomous agent.

Wraps the GitHub REST API so the agent can read repository contents,
create branches, commit files, open pull requests, and manage issues.

Authentication uses a GitHub App (recommended) or a Personal Access Token.
GitHub App credentials are loaded from environment variables:
  - GITHUB_APP_ID
  - GITHUB_APP_PRIVATE_KEY_PATH
  - GITHUB_APP_INSTALLATION_ID

Alternatively, a simple PAT can be provided via GITHUB_TOKEN.

The integration is fully optional — if no credentials are found, no tools
are registered and no errors are raised.
"""

from __future__ import annotations

import json
import os
import time
from base64 import b64decode
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from klovis_agent.perception.base import Event, EventKind, PerceptionSource
from klovis_agent.tools.base import BaseTool, ToolResult, ToolSpec

if TYPE_CHECKING:
    from klovis_agent.tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)

_BASE = "https://api.github.com"
_TIMEOUT = 30
_MAX_RESPONSE_CHARS = 80_000


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def _load_github_config() -> dict[str, str]:
    """Load GitHub credentials from environment.

    Returns a dict with at least ``token`` set if credentials are available,
    or an empty dict otherwise.  For GitHub App auth the dict also contains
    ``app_id``, ``installation_id``, and ``private_key_path``.
    """
    pat = os.environ.get("GITHUB_TOKEN", "")
    if pat:
        return {"token": pat, "auth_mode": "pat"}

    app_id = os.environ.get("GITHUB_APP_ID", "")
    key_path = os.environ.get("GITHUB_APP_PRIVATE_KEY_PATH", "")
    install_id = os.environ.get("GITHUB_APP_INSTALLATION_ID", "")

    if app_id and key_path and install_id:
        resolved = Path(key_path).expanduser().resolve()
        if resolved.is_file():
            return {
                "app_id": app_id,
                "private_key_path": str(resolved),
                "installation_id": install_id,
                "auth_mode": "app",
            }
        logger.warning("github_private_key_not_found", path=key_path)

    return {}


def _generate_jwt(app_id: str, private_key_pem: str) -> str:
    """Create a short-lived JWT for GitHub App authentication.

    Tries, in order: PyJWT, cryptography, then the system ``openssl``
    CLI.  This ensures GitHub App auth works even without any optional
    Python crypto dependency installed.
    """
    import base64

    now = int(time.time())
    payload_dict = {"iat": now - 60, "exp": now + 540, "iss": app_id}

    try:
        import jwt

        return jwt.encode(payload_dict, private_key_pem, algorithm="RS256")
    except ImportError:
        pass

    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding

        header = base64.urlsafe_b64encode(
            json.dumps({"alg": "RS256", "typ": "JWT"}).encode()
        ).rstrip(b"=")
        body = base64.urlsafe_b64encode(
            json.dumps(payload_dict).encode()
        ).rstrip(b"=")
        signing_input = header + b"." + body

        key = serialization.load_pem_private_key(
            private_key_pem.encode(), password=None,
        )
        signature = key.sign(  # type: ignore[union-attr]
            signing_input, padding.PKCS1v15(), hashes.SHA256(),
        )
        sig_b64 = base64.urlsafe_b64encode(signature).rstrip(b"=")
        return (signing_input + b"." + sig_b64).decode()
    except ImportError:
        pass

    return _generate_jwt_openssl(app_id, payload_dict)


def _generate_jwt_openssl(app_id: str, payload_dict: dict) -> str:
    """Sign a JWT RS256 using the system ``openssl`` command."""
    import base64
    import subprocess
    import tempfile

    header = base64.urlsafe_b64encode(
        json.dumps({"alg": "RS256", "typ": "JWT"}).encode()
    ).rstrip(b"=")
    body = base64.urlsafe_b64encode(
        json.dumps(payload_dict).encode()
    ).rstrip(b"=")
    signing_input = header + b"." + body

    key_path = os.environ.get("GITHUB_APP_PRIVATE_KEY_PATH", "")
    resolved_key = str(Path(key_path).expanduser().resolve()) if key_path else ""

    if not resolved_key or not Path(resolved_key).is_file():
        raise RuntimeError(
            "GitHub App JWT signing requires PyJWT, cryptography, "
            "or the openssl CLI with GITHUB_APP_PRIVATE_KEY_PATH set"
        )

    with tempfile.NamedTemporaryFile(suffix=".bin") as sig_file:
        proc = subprocess.run(
            [
                "openssl", "dgst", "-sha256", "-sign", resolved_key,
                "-out", sig_file.name,
            ],
            input=signing_input,
            capture_output=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"openssl signing failed: {proc.stderr.decode()}"
            )
        signature = Path(sig_file.name).read_bytes()

    sig_b64 = base64.urlsafe_b64encode(signature).rstrip(b"=")
    return (signing_input + b"." + sig_b64).decode()


class _GitHubAuth:
    """Manages GitHub authentication, handling token refresh for App auth."""

    def __init__(self, config: dict[str, str]) -> None:
        self._config = config
        self._token: str = config.get("token", "")
        self._token_expires_at: float = 0.0

    @property
    def is_app(self) -> bool:
        return self._config.get("auth_mode") == "app"

    async def get_token(self) -> str:
        if not self.is_app:
            return self._token

        if self._token and time.time() < self._token_expires_at - 60:
            return self._token

        private_key = Path(self._config["private_key_path"]).read_text()
        jwt_token = _generate_jwt(self._config["app_id"], private_key)

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(
                f"{_BASE}/app/installations/{self._config['installation_id']}/access_tokens",
                headers={
                    "Authorization": f"Bearer {jwt_token}",
                    "Accept": "application/vnd.github+json",
                },
            )
            if resp.status_code >= 400:
                logger.error("github_token_refresh_failed", status=resp.status_code)
                raise RuntimeError(f"GitHub App token refresh failed: {resp.status_code}")

            data = resp.json()
            self._token = data["token"]
            from datetime import datetime
            expires = data.get("expires_at", "")
            if expires:
                dt = datetime.fromisoformat(expires.replace("Z", "+00:00"))
                self._token_expires_at = dt.timestamp()
            else:
                self._token_expires_at = time.time() + 3500

        logger.info("github_app_token_refreshed")
        return self._token

    async def headers(self) -> dict[str, str]:
        token = await self.get_token()
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }


def _safe_json(resp: httpx.Response) -> dict[str, Any]:
    try:
        data = resp.json()
        if isinstance(data, list):
            return {"items": data}
        return data  # type: ignore[no-any-return]
    except Exception:
        return {"raw": resp.text[:_MAX_RESPONSE_CHARS]}


def _truncate(text: str, max_chars: int = _MAX_RESPONSE_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... (truncated, {len(text)} total chars)"


def _parse_owner_repo(inputs: dict[str, Any]) -> tuple[str, str]:
    """Extract and normalise ``owner`` and ``repo`` from tool inputs.

    LLMs frequently pass the full slug (``"owner/repo"``) or even a full
    GitHub URL as the ``owner`` value.  This helper handles all variants
    so individual tools don't have to.
    """
    raw_owner = inputs.get("owner") or ""
    raw_repo = inputs.get("repo") or ""
    owner = raw_owner.strip().rstrip("/")
    repo = raw_repo.strip().strip("/")

    if "github.com" in owner:
        parts = owner.split("github.com")[-1].strip("/").split("/")
        if len(parts) >= 2:
            owner, repo = parts[0], parts[1]
        elif len(parts) == 1:
            owner = parts[0]

    if "/" in owner:
        segments = owner.split("/")
        owner = segments[0]
        if not repo:
            repo = segments[1]

    if "github.com" in repo:
        parts = repo.split("github.com")[-1].strip("/").split("/")
        if len(parts) >= 2:
            owner, repo = parts[0], parts[1]
        elif len(parts) == 1:
            repo = parts[0]

    if raw_owner != owner or raw_repo != repo:
        logger.info(
            "github_owner_repo_normalised",
            raw_owner=raw_owner, raw_repo=raw_repo,
            owner=owner, repo=repo,
        )

    return owner, repo


_OWNER_DESC = (
    "The GitHub username or organisation that owns the repository. "
    "Must be the username only — do NOT include the repo name or a URL."
)
_REPO_DESC = (
    "The repository name. "
    "Must be the repo name only — do NOT include the owner or a URL."
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class GitHubGetRepoTool(BaseTool):
    """Get repository metadata."""

    def __init__(self, auth: _GitHubAuth) -> None:
        super().__init__()
        self._auth = auth

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="github_get_repo",
            description=(
                "Get metadata about a GitHub repository (description, default branch, "
                "open issues count, language, etc.)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": _OWNER_DESC},
                    "repo": {"type": "string", "description": _REPO_DESC},
                },
                "required": ["owner", "repo"],
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        owner, repo = _parse_owner_repo(inputs)
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(
                    f"{_BASE}/repos/{owner}/{repo}",
                    headers=await self._auth.headers(),
                )
            data = _safe_json(resp)
            return ToolResult(
                success=resp.status_code < 400,
                output=data,
                error=data.get("message") if resp.status_code >= 400 else None,
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class GitHubReadFileTool(BaseTool):
    """Read a file from a GitHub repository."""

    def __init__(self, auth: _GitHubAuth) -> None:
        super().__init__()
        self._auth = auth

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="github_read_file",
            description=(
                "Read the contents of a file from a GitHub repository at a given "
                "ref (branch, tag, or commit SHA). Returns the decoded text content."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": _OWNER_DESC},
                    "repo": {"type": "string", "description": _REPO_DESC},
                    "path": {"type": "string", "description": "File path in the repo"},
                    "ref": {
                        "type": "string",
                        "description": "Branch, tag, or SHA (defaults to repo default branch)",
                    },
                },
                "required": ["owner", "repo", "path"],
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        owner, repo = _parse_owner_repo(inputs)
        params: dict[str, str] = {}
        if ref := inputs.get("ref"):
            params["ref"] = ref

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(
                    f"{_BASE}/repos/{owner}/{repo}/contents/{inputs['path']}",
                    headers=await self._auth.headers(),
                    params=params,
                )
            if resp.status_code >= 400:
                data = _safe_json(resp)
                return ToolResult(
                    success=False, output=data,
                    error=data.get("message", f"HTTP {resp.status_code}"),
                )

            data = resp.json()
            if isinstance(data, list):
                entries = [
                    {"name": e["name"], "type": e["type"], "path": e["path"]}
                    for e in data
                ]
                return ToolResult(success=True, output={"type": "directory", "entries": entries})

            if data.get("encoding") == "base64" and data.get("content"):
                content = b64decode(data["content"]).decode("utf-8", errors="replace")
                return ToolResult(success=True, output={
                    "path": data.get("path", inputs["path"]),
                    "sha": data.get("sha", ""),
                    "size": data.get("size", 0),
                    "content": _truncate(content),
                })

            return ToolResult(success=True, output=data)
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class GitHubListFilesTool(BaseTool):
    """List files in a repository directory via the Git Trees API."""

    def __init__(self, auth: _GitHubAuth) -> None:
        super().__init__()
        self._auth = auth

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="github_list_files",
            description=(
                "List files and directories in a GitHub repository path. "
                "Use recursive=true to get the full file tree."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": _OWNER_DESC},
                    "repo": {"type": "string", "description": _REPO_DESC},
                    "path": {
                        "type": "string",
                        "description": "Directory path (empty string for root)",
                    },
                    "ref": {
                        "type": "string",
                        "description": "Branch, tag, or SHA (defaults to repo default branch)",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "List recursively (default false)",
                    },
                },
                "required": ["owner", "repo"],
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        owner, repo = _parse_owner_repo(inputs)
        path = inputs.get("path", "")
        params: dict[str, str] = {}
        if ref := inputs.get("ref"):
            params["ref"] = ref

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(
                    f"{_BASE}/repos/{owner}/{repo}/contents/{path}",
                    headers=await self._auth.headers(),
                    params=params,
                )
            if resp.status_code >= 400:
                data = _safe_json(resp)
                return ToolResult(
                    success=False, output=data,
                    error=data.get("message", f"HTTP {resp.status_code}"),
                )

            data = resp.json()
            if isinstance(data, list):
                entries = [
                    {
                        "name": e["name"], "type": e["type"],
                        "path": e["path"], "size": e.get("size", 0),
                    }
                    for e in data
                ]
                return ToolResult(success=True, output={"entries": entries, "count": len(entries)})

            return ToolResult(success=True, output={"entries": [data], "count": 1})
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class GitHubCreateBranchTool(BaseTool):
    """Create a new branch in a repository."""

    requires_confirmation = True

    def __init__(self, auth: _GitHubAuth, **kw: Any) -> None:
        super().__init__(**kw)
        self._auth = auth

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="github_create_branch",
            description=(
                "Create a new branch in a GitHub repository from a base ref. "
                "The base ref can be a branch name, tag, or commit SHA."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": _OWNER_DESC},
                    "repo": {"type": "string", "description": _REPO_DESC},
                    "branch": {"type": "string", "description": "New branch name"},
                    "from_ref": {
                        "type": "string",
                        "description": (
                            "Base branch or SHA to branch from "
                            "(defaults to repo default branch)"
                        ),
                    },
                },
                "required": ["owner", "repo", "branch"],
            },
        )

    def describe_action(self, inputs: dict[str, Any]) -> str:
        owner, repo = _parse_owner_repo(inputs)
        branch = inputs.get("branch")
        return f"Create branch '{branch}' in {owner}/{repo}"

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        owner, repo = _parse_owner_repo(inputs)
        branch = inputs["branch"]
        from_ref = inputs.get("from_ref", "")

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                headers = await self._auth.headers()

                if not from_ref:
                    resp = await client.get(
                        f"{_BASE}/repos/{owner}/{repo}", headers=headers,
                    )
                    if resp.status_code >= 400:
                        return ToolResult(
                        success=False,
                        error=f"Cannot fetch repo: HTTP {resp.status_code}",
                    )
                    from_ref = resp.json().get("default_branch", "main")

                resp = await client.get(
                    f"{_BASE}/repos/{owner}/{repo}/git/ref/heads/{from_ref}",
                    headers=headers,
                )
                if resp.status_code >= 400:
                    return ToolResult(
                        success=False,
                        error=f"Cannot resolve ref '{from_ref}': HTTP {resp.status_code}",
                    )
                sha = resp.json()["object"]["sha"]

                resp = await client.post(
                    f"{_BASE}/repos/{owner}/{repo}/git/refs",
                    headers=headers,
                    json={"ref": f"refs/heads/{branch}", "sha": sha},
                )

            data = _safe_json(resp)
            if resp.status_code >= 400:
                return ToolResult(
                    success=False, output=data,
                    error=data.get("message", f"HTTP {resp.status_code}"),
                )
            return ToolResult(success=True, output={
                "branch": branch, "sha": sha, "ref": data.get("ref", ""),
            })
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class GitHubCommitFilesTool(BaseTool):
    """Create a commit with one or more file changes via the Git Trees API."""

    requires_confirmation = True

    def __init__(self, auth: _GitHubAuth, **kw: Any) -> None:
        super().__init__(**kw)
        self._auth = auth

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="github_commit_files",
            description=(
                "Create a Git commit with one or more file changes on a branch. "
                "Uses the Git Trees API — no local clone needed. Each file change "
                "specifies a path and the new content. Files can be created, updated, "
                "or deleted (set content to null to delete)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": _OWNER_DESC},
                    "repo": {"type": "string", "description": _REPO_DESC},
                    "branch": {"type": "string", "description": "Branch to commit to"},
                    "message": {"type": "string", "description": "Commit message"},
                    "files": {
                        "type": "array",
                        "description": "List of file changes",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "File path in the repo"},
                                "content": {
                                    "type": ["string", "null"],
                                    "description": "New file content (null to delete the file)",
                                },
                            },
                            "required": ["path"],
                        },
                    },
                },
                "required": ["owner", "repo", "branch", "message", "files"],
            },
        )

    def describe_action(self, inputs: dict[str, Any]) -> str:
        owner, repo = _parse_owner_repo(inputs)
        n = len(inputs.get("files", []))
        branch = inputs.get("branch")
        return f"Commit {n} file(s) to {owner}/{repo}:{branch}"

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        owner, repo = _parse_owner_repo(inputs)
        branch = inputs["branch"]
        message = inputs["message"]
        files = inputs["files"]

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                headers = await self._auth.headers()

                resp = await client.get(
                    f"{_BASE}/repos/{owner}/{repo}/git/ref/heads/{branch}",
                    headers=headers,
                )
                if resp.status_code >= 400:
                    return ToolResult(success=False, error=f"Branch '{branch}' not found")
                base_sha = resp.json()["object"]["sha"]

                resp = await client.get(
                    f"{_BASE}/repos/{owner}/{repo}/git/commits/{base_sha}",
                    headers=headers,
                )
                base_tree_sha = resp.json()["tree"]["sha"]

                tree_items = []
                for f in files:
                    if f.get("content") is None:
                        tree_items.append({
                            "path": f["path"],
                            "mode": "100644",
                            "type": "blob",
                            "sha": None,
                        })
                    else:
                        tree_items.append({
                            "path": f["path"],
                            "mode": "100644",
                            "type": "blob",
                            "content": f["content"],
                        })

                resp = await client.post(
                    f"{_BASE}/repos/{owner}/{repo}/git/trees",
                    headers=headers,
                    json={"base_tree": base_tree_sha, "tree": tree_items},
                )
                if resp.status_code >= 400:
                    data = _safe_json(resp)
                    return ToolResult(
                        success=False, output=data,
                        error=data.get("message", "Tree creation failed"),
                    )
                new_tree_sha = resp.json()["sha"]

                resp = await client.post(
                    f"{_BASE}/repos/{owner}/{repo}/git/commits",
                    headers=headers,
                    json={
                        "message": message,
                        "tree": new_tree_sha,
                        "parents": [base_sha],
                    },
                )
                if resp.status_code >= 400:
                    data = _safe_json(resp)
                    return ToolResult(
                        success=False, output=data,
                        error=data.get("message", "Commit creation failed"),
                    )
                new_commit_sha = resp.json()["sha"]

                resp = await client.patch(
                    f"{_BASE}/repos/{owner}/{repo}/git/refs/heads/{branch}",
                    headers=headers,
                    json={"sha": new_commit_sha},
                )
                if resp.status_code >= 400:
                    data = _safe_json(resp)
                    return ToolResult(
                        success=False, output=data,
                        error=data.get("message", "Ref update failed"),
                    )

            return ToolResult(success=True, output={
                "commit_sha": new_commit_sha,
                "branch": branch,
                "files_changed": len(files),
                "message": message,
            })
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class GitHubCreatePRTool(BaseTool):
    """Open a pull request."""

    requires_confirmation = True

    def __init__(self, auth: _GitHubAuth, **kw: Any) -> None:
        super().__init__(**kw)
        self._auth = auth

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="github_create_pr",
            description=(
                "Create a pull request on a GitHub repository. Returns the PR "
                "number and URL."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": _OWNER_DESC},
                    "repo": {"type": "string", "description": _REPO_DESC},
                    "title": {"type": "string", "description": "PR title"},
                    "body": {"type": "string", "description": "PR description (markdown)"},
                    "head": {"type": "string", "description": "Branch with changes"},
                    "base": {
                        "type": "string",
                        "description": "Target branch (defaults to repo default branch)",
                    },
                    "draft": {
                        "type": "boolean",
                        "description": "Create as draft PR (default false)",
                    },
                },
                "required": ["owner", "repo", "title", "head"],
            },
        )

    def describe_action(self, inputs: dict[str, Any]) -> str:
        owner, repo = _parse_owner_repo(inputs)
        return f"Open PR '{inputs.get('title')}' in {owner}/{repo}"

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        owner, repo = _parse_owner_repo(inputs)
        payload: dict[str, Any] = {
            "title": inputs["title"],
            "head": inputs["head"],
            "base": inputs.get("base", "main"),
        }
        if body := inputs.get("body"):
            payload["body"] = body
        if inputs.get("draft"):
            payload["draft"] = True

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.post(
                    f"{_BASE}/repos/{owner}/{repo}/pulls",
                    headers=await self._auth.headers(),
                    json=payload,
                )
            data = _safe_json(resp)
            if resp.status_code >= 400:
                return ToolResult(
                    success=False, output=data,
                    error=data.get("message", f"HTTP {resp.status_code}"),
                )
            return ToolResult(success=True, output={
                "number": data.get("number"),
                "url": data.get("html_url", ""),
                "state": data.get("state", ""),
                "title": data.get("title", ""),
                "head": data.get("head", {}).get("ref", ""),
                "base": data.get("base", {}).get("ref", ""),
            })
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class GitHubListIssuesTool(BaseTool):
    """List issues on a repository."""

    def __init__(self, auth: _GitHubAuth) -> None:
        super().__init__()
        self._auth = auth

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="github_list_issues",
            description=(
                "List issues on a GitHub repository. Supports filtering by state, "
                "labels, assignee, and sorting."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": _OWNER_DESC},
                    "repo": {"type": "string", "description": _REPO_DESC},
                    "state": {"type": "string", "enum": ["open", "closed", "all"]},
                    "labels": {"type": "string", "description": "Comma-separated label names"},
                    "assignee": {"type": "string"},
                    "sort": {"type": "string", "enum": ["created", "updated", "comments"]},
                    "direction": {"type": "string", "enum": ["asc", "desc"]},
                    "per_page": {"type": "integer", "description": "Results per page (max 100)"},
                    "page": {"type": "integer"},
                },
                "required": ["owner", "repo"],
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        owner, repo = _parse_owner_repo(inputs)
        params: dict[str, Any] = {}
        for key in ("state", "labels", "assignee", "sort", "direction", "per_page", "page"):
            if val := inputs.get(key):
                params[key] = val

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(
                    f"{_BASE}/repos/{owner}/{repo}/issues",
                    headers=await self._auth.headers(),
                    params=params,
                )
            if resp.status_code >= 400:
                data = _safe_json(resp)
                return ToolResult(
                    success=False, output=data,
                    error=data.get("message", f"HTTP {resp.status_code}"),
                )

            raw = resp.json()
            issues = [
                {
                    "number": i["number"],
                    "title": i["title"],
                    "state": i["state"],
                    "labels": [lb["name"] for lb in i.get("labels", [])],
                    "assignee": (i.get("assignee") or {}).get("login", ""),
                    "created_at": i.get("created_at", ""),
                    "updated_at": i.get("updated_at", ""),
                    "is_pull_request": "pull_request" in i,
                    "url": i.get("html_url", ""),
                }
                for i in (raw if isinstance(raw, list) else [])
            ]
            return ToolResult(success=True, output={"issues": issues, "count": len(issues)})
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class GitHubListPRsTool(BaseTool):
    """List pull requests on a repository."""

    def __init__(self, auth: _GitHubAuth) -> None:
        super().__init__()
        self._auth = auth

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="github_list_prs",
            description="List pull requests on a GitHub repository.",
            input_schema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": _OWNER_DESC},
                    "repo": {"type": "string", "description": _REPO_DESC},
                    "state": {"type": "string", "enum": ["open", "closed", "all"]},
                    "sort": {
                        "type": "string",
                        "enum": ["created", "updated", "popularity", "long-running"],
                    },
                    "direction": {"type": "string", "enum": ["asc", "desc"]},
                    "per_page": {"type": "integer"},
                    "page": {"type": "integer"},
                },
                "required": ["owner", "repo"],
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        owner, repo = _parse_owner_repo(inputs)
        params: dict[str, Any] = {}
        for key in ("state", "sort", "direction", "per_page", "page"):
            if val := inputs.get(key):
                params[key] = val

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(
                    f"{_BASE}/repos/{owner}/{repo}/pulls",
                    headers=await self._auth.headers(),
                    params=params,
                )
            if resp.status_code >= 400:
                data = _safe_json(resp)
                return ToolResult(
                    success=False, output=data,
                    error=data.get("message", f"HTTP {resp.status_code}"),
                )

            raw = resp.json()
            prs = [
                {
                    "number": p["number"],
                    "title": p["title"],
                    "state": p["state"],
                    "draft": p.get("draft", False),
                    "head": p.get("head", {}).get("ref", ""),
                    "base": p.get("base", {}).get("ref", ""),
                    "user": p.get("user", {}).get("login", ""),
                    "created_at": p.get("created_at", ""),
                    "updated_at": p.get("updated_at", ""),
                    "url": p.get("html_url", ""),
                }
                for p in (raw if isinstance(raw, list) else [])
            ]
            return ToolResult(success=True, output={"pull_requests": prs, "count": len(prs)})
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class GitHubGetPRTool(BaseTool):
    """Get details and diff of a specific pull request."""

    def __init__(self, auth: _GitHubAuth) -> None:
        super().__init__()
        self._auth = auth

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="github_get_pr",
            description=(
                "Get detailed information about a pull request, including its "
                "diff, review comments, and merge status."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": _OWNER_DESC},
                    "repo": {"type": "string", "description": _REPO_DESC},
                    "pr_number": {"type": "integer", "description": "Pull request number"},
                    "include_diff": {
                        "type": "boolean",
                        "description": "Include the full diff (default false)",
                    },
                },
                "required": ["owner", "repo", "pr_number"],
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        owner, repo = _parse_owner_repo(inputs)
        pr_number = inputs["pr_number"]

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                headers = await self._auth.headers()

                resp = await client.get(
                    f"{_BASE}/repos/{owner}/{repo}/pulls/{pr_number}",
                    headers=headers,
                )
                if resp.status_code >= 400:
                    data = _safe_json(resp)
                    return ToolResult(success=False, output=data, error=data.get("message"))

                pr = resp.json()
                result: dict[str, Any] = {
                    "number": pr["number"],
                    "title": pr["title"],
                    "state": pr["state"],
                    "body": _truncate(pr.get("body", "") or "", 5000),
                    "draft": pr.get("draft", False),
                    "mergeable": pr.get("mergeable"),
                    "head": pr.get("head", {}).get("ref", ""),
                    "base": pr.get("base", {}).get("ref", ""),
                    "user": pr.get("user", {}).get("login", ""),
                    "changed_files": pr.get("changed_files", 0),
                    "additions": pr.get("additions", 0),
                    "deletions": pr.get("deletions", 0),
                    "url": pr.get("html_url", ""),
                }

                if inputs.get("include_diff"):
                    diff_headers = {**headers, "Accept": "application/vnd.github.diff"}
                    resp = await client.get(
                        f"{_BASE}/repos/{owner}/{repo}/pulls/{pr_number}",
                        headers=diff_headers,
                    )
                    if resp.status_code < 400:
                        result["diff"] = _truncate(resp.text)

                resp = await client.get(
                    f"{_BASE}/repos/{owner}/{repo}/pulls/{pr_number}/reviews",
                    headers=headers,
                )
                if resp.status_code < 400:
                    reviews = resp.json()
                    result["reviews"] = [
                        {
                            "user": r.get("user", {}).get("login", ""),
                            "state": r.get("state", ""),
                            "body": (r.get("body", "") or "")[:500],
                        }
                        for r in (reviews if isinstance(reviews, list) else [])
                    ]

            return ToolResult(success=True, output=result)
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class GitHubSearchCodeTool(BaseTool):
    """Search code across repositories."""

    def __init__(self, auth: _GitHubAuth) -> None:
        super().__init__()
        self._auth = auth

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="github_search_code",
            description=(
                "Search for code in a GitHub repository. Useful for finding "
                "specific patterns, function definitions, or usages across files."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search query. Use GitHub search syntax: "
                            "'def my_func repo:owner/repo' or 'import xyz path:src/'"
                        ),
                    },
                    "per_page": {"type": "integer", "description": "Results per page (max 100)"},
                    "page": {"type": "integer"},
                },
                "required": ["query"],
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        params: dict[str, Any] = {"q": inputs["query"]}
        for key in ("per_page", "page"):
            if val := inputs.get(key):
                params[key] = val

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(
                    f"{_BASE}/search/code",
                    headers=await self._auth.headers(),
                    params=params,
                )
            if resp.status_code >= 400:
                data = _safe_json(resp)
                return ToolResult(
                    success=False, output=data,
                    error=data.get("message", f"HTTP {resp.status_code}"),
                )

            data = resp.json()
            results = [
                {
                    "name": item["name"],
                    "path": item["path"],
                    "repo": item.get("repository", {}).get("full_name", ""),
                    "url": item.get("html_url", ""),
                    "score": item.get("score", 0),
                }
                for item in data.get("items", [])
            ]
            return ToolResult(success=True, output={
                "total_count": data.get("total_count", 0),
                "results": results,
            })
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class GitHubCloneRepoTool(BaseTool):
    """Clone a GitHub repository to a local directory for development."""

    requires_confirmation = True

    def __init__(self, auth: _GitHubAuth, scratch_dir: Path, **kw: Any) -> None:
        super().__init__(**kw)
        self._auth = auth
        self._scratch_dir = scratch_dir

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="github_clone_repo",
            description=(
                "Clone a GitHub repository to a local directory. "
                "Use this to work on code locally: edit files, run tests, "
                "run linters, then commit and push. The clone is "
                "authenticated so you can push changes back. "
                "Returns the local path where the repo was cloned."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": _OWNER_DESC},
                    "repo": {"type": "string", "description": _REPO_DESC},
                    "branch": {
                        "type": "string",
                        "description": (
                            "Branch to checkout after cloning "
                            "(defaults to the repo default branch)"
                        ),
                    },
                },
                "required": ["owner", "repo"],
            },
        )

    def describe_action(self, inputs: dict[str, Any]) -> str:
        owner, repo = _parse_owner_repo(inputs)
        return f"Clone {owner}/{repo} locally"

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        import asyncio as _asyncio

        owner, repo = _parse_owner_repo(inputs)
        branch = inputs.get("branch", "")

        clone_dir = self._scratch_dir / repo
        if clone_dir.exists():
            return ToolResult(success=True, output={
                "path": str(clone_dir),
                "already_existed": True,
                "message": f"Repository already cloned at {clone_dir}",
            })

        token = await self._auth.get_token()
        clone_url = f"https://x-access-token:{token}@github.com/{owner}/{repo}.git"

        cmd = ["git", "clone", "--depth", "50", clone_url, str(clone_dir)]
        try:
            proc = await _asyncio.create_subprocess_exec(
                *cmd,
                stdout=_asyncio.subprocess.PIPE,
                stderr=_asyncio.subprocess.PIPE,
            )
            stdout, stderr = await _asyncio.wait_for(
                proc.communicate(), timeout=120,
            )

            if proc.returncode != 0:
                err = stderr.decode("utf-8", errors="replace")
                err_safe = err.replace(token, "***")
                return ToolResult(success=False, error=f"git clone failed: {err_safe}")

            if branch:
                proc2 = await _asyncio.create_subprocess_exec(
                    "git", "checkout", branch,
                    cwd=str(clone_dir),
                    stdout=_asyncio.subprocess.PIPE,
                    stderr=_asyncio.subprocess.PIPE,
                )
                await _asyncio.wait_for(proc2.communicate(), timeout=30)

            logger.info(
                "github_repo_cloned",
                owner=owner, repo=repo, path=str(clone_dir),
            )
            return ToolResult(success=True, output={
                "path": str(clone_dir),
                "owner": owner,
                "repo": repo,
                "branch": branch or "(default)",
                "message": (
                    f"Repository cloned to {clone_dir}. "
                    f"Use shell_command with cwd='{clone_dir}' to run "
                    f"git, pytest, ruff, etc. Use file_read/file_edit "
                    f"with paths under '{clone_dir}' to modify files."
                ),
            })

        except TimeoutError:
            return ToolResult(success=False, error="git clone timed out (120s)")
        except Exception as exc:
            err_safe = str(exc).replace(token, "***")
            return ToolResult(success=False, error=f"Clone error: {err_safe}")


class GitHubCreateIssueTool(BaseTool):
    """Create an issue on a GitHub repository."""

    requires_confirmation = True

    def __init__(self, auth: _GitHubAuth, **kw: Any) -> None:
        super().__init__(**kw)
        self._auth = auth

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="github_create_issue",
            description=(
                "Create an issue on a GitHub repository. "
                "Useful for tracking work items, bugs, or improvement ideas."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": _OWNER_DESC},
                    "repo": {"type": "string", "description": _REPO_DESC},
                    "title": {"type": "string", "description": "Issue title"},
                    "body": {
                        "type": "string",
                        "description": "Issue body (markdown)",
                    },
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Labels to apply (e.g. ['bug', 'agent'])",
                    },
                    "assignees": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "GitHub usernames to assign",
                    },
                },
                "required": ["owner", "repo", "title"],
            },
        )

    def describe_action(self, inputs: dict[str, Any]) -> str:
        owner, repo = _parse_owner_repo(inputs)
        return f"Create issue '{inputs.get('title')}' in {owner}/{repo}"

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        owner, repo = _parse_owner_repo(inputs)
        payload: dict[str, Any] = {"title": inputs["title"]}
        if body := inputs.get("body"):
            payload["body"] = body
        if labels := inputs.get("labels"):
            payload["labels"] = labels
        if assignees := inputs.get("assignees"):
            payload["assignees"] = assignees

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.post(
                    f"{_BASE}/repos/{owner}/{repo}/issues",
                    headers=await self._auth.headers(),
                    json=payload,
                )
            data = _safe_json(resp)
            if resp.status_code >= 400:
                return ToolResult(
                    success=False, output=data,
                    error=data.get("message", f"HTTP {resp.status_code}"),
                )
            return ToolResult(success=True, output={
                "number": data.get("number"),
                "url": data.get("html_url", ""),
                "state": data.get("state", ""),
                "title": data.get("title", ""),
                "labels": [lb.get("name", "") for lb in data.get("labels", [])],
            })
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class GitHubCommentIssueTool(BaseTool):
    """Add a comment to an issue or pull request."""

    requires_confirmation = True

    def __init__(self, auth: _GitHubAuth, **kw: Any) -> None:
        super().__init__(**kw)
        self._auth = auth

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="github_comment_issue",
            description=(
                "Add a comment to a GitHub issue or pull request. "
                "Useful for reporting progress, asking questions, or leaving notes."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": _OWNER_DESC},
                    "repo": {"type": "string", "description": _REPO_DESC},
                    "issue_number": {
                        "type": "integer",
                        "description": "Issue or PR number",
                    },
                    "body": {
                        "type": "string",
                        "description": "Comment body (markdown)",
                    },
                },
                "required": ["owner", "repo", "issue_number", "body"],
            },
        )

    def describe_action(self, inputs: dict[str, Any]) -> str:
        owner, repo = _parse_owner_repo(inputs)
        num = inputs.get("issue_number")
        return f"Comment on {owner}/{repo}#{num}"

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        owner, repo = _parse_owner_repo(inputs)
        issue_number = inputs["issue_number"]

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.post(
                    f"{_BASE}/repos/{owner}/{repo}/issues/{issue_number}/comments",
                    headers=await self._auth.headers(),
                    json={"body": inputs["body"]},
                )
            data = _safe_json(resp)
            if resp.status_code >= 400:
                return ToolResult(
                    success=False, output=data,
                    error=data.get("message", f"HTTP {resp.status_code}"),
                )
            return ToolResult(success=True, output={
                "id": data.get("id"),
                "url": data.get("html_url", ""),
                "created_at": data.get("created_at", ""),
            })
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class GitHubGetCheckRunsTool(BaseTool):
    """Get CI check-run results for a commit or branch."""

    def __init__(self, auth: _GitHubAuth) -> None:
        super().__init__()
        self._auth = auth

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="github_get_check_runs",
            description=(
                "Get CI/CD check-run results for a given commit SHA or branch. "
                "Returns the status and conclusion of each check."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": _OWNER_DESC},
                    "repo": {"type": "string", "description": _REPO_DESC},
                    "ref": {
                        "type": "string",
                        "description": "Commit SHA or branch name",
                    },
                },
                "required": ["owner", "repo", "ref"],
            },
        )

    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        owner, repo = _parse_owner_repo(inputs)
        ref = inputs["ref"]

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(
                    f"{_BASE}/repos/{owner}/{repo}/commits/{ref}/check-runs",
                    headers=await self._auth.headers(),
                )
            if resp.status_code >= 400:
                data = _safe_json(resp)
                return ToolResult(
                    success=False, output=data,
                    error=data.get("message", f"HTTP {resp.status_code}"),
                )

            data = resp.json()
            runs = [
                {
                    "name": cr.get("name", ""),
                    "status": cr.get("status", ""),
                    "conclusion": cr.get("conclusion"),
                    "started_at": cr.get("started_at", ""),
                    "completed_at": cr.get("completed_at", ""),
                    "html_url": cr.get("html_url", ""),
                }
                for cr in data.get("check_runs", [])
            ]
            return ToolResult(success=True, output={
                "total_count": data.get("total_count", 0),
                "check_runs": runs,
            })
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


# ---------------------------------------------------------------------------
# Perception source
# ---------------------------------------------------------------------------


class GitHubPerceptionSource(PerceptionSource):
    """Polls GitHub for repository events the agent should be aware of.

    Tracks notification IDs and issue numbers already seen within the
    process lifetime to avoid re-surfacing the same events across daemon
    cycles.

    When ``issue_labels`` is set, the source also polls for open issues
    matching those labels — this lets the agent pick up work items
    (e.g. issues labelled ``agent``) without relying on the notification
    stream.
    """

    def __init__(
        self,
        owner: str,
        repo: str,
        auth: _GitHubAuth | None = None,
        *,
        issue_labels: list[str] | None = None,
    ) -> None:
        self._owner = owner
        self._repo = repo
        self._auth = auth
        self._issue_labels = issue_labels or []
        self._seen_notification_ids: set[str] = set()
        self._seen_issue_numbers: set[int] = set()

    @property
    def name(self) -> str:
        return "github"

    async def poll(self) -> list[Event]:
        if not self._auth:
            return []

        events: list[Event] = []

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                headers = await self._auth.headers()

                resp = await client.get(
                    f"{_BASE}/repos/{self._owner}/{self._repo}/notifications",
                    headers=headers,
                    params={"all": "false", "per_page": "20"},
                )
                if resp.status_code < 400:
                    notifications = resp.json()
                    for n in (notifications if isinstance(notifications, list) else []):
                        notif_id = str(n.get("id", ""))
                        if notif_id in self._seen_notification_ids:
                            continue
                        if notif_id:
                            self._seen_notification_ids.add(notif_id)

                        subject = n.get("subject", {})
                        reason = n.get("reason", "")
                        kind_map = {
                            "mention": EventKind.MENTION,
                            "review_requested": EventKind.REQUEST,
                            "comment": EventKind.NOTIFICATION,
                            "ci_activity": EventKind.SYSTEM,
                        }
                        events.append(Event(
                            source="github",
                            kind=kind_map.get(reason, EventKind.NOTIFICATION),
                            title=subject.get("title", "GitHub notification"),
                            detail=f"[{subject.get('type', '')}] {reason}",
                            metadata={
                                "notification_id": notif_id,
                                "reason": reason,
                                "subject_type": subject.get("type", ""),
                                "subject_url": subject.get("url", ""),
                                "repo": f"{self._owner}/{self._repo}",
                            },
                        ))

                resp = await client.get(
                    f"{_BASE}/repos/{self._owner}/{self._repo}/events",
                    headers=headers,
                    params={"per_page": "10"},
                )
                if resp.status_code < 400:
                    repo_events = resp.json()
                    for ev in (repo_events if isinstance(repo_events, list) else []):
                        ev_type = ev.get("type", "")
                        if ev_type in ("IssuesEvent", "PullRequestEvent", "PushEvent"):
                            payload = ev.get("payload", {})
                            action = payload.get("action", "")
                            title = ""
                            if ev_type == "IssuesEvent":
                                issue_title = payload.get("issue", {}).get("title", "")
                                title = f"Issue {action}: {issue_title}"
                            elif ev_type == "PullRequestEvent":
                                pr_title = payload.get("pull_request", {}).get("title", "")
                                title = f"PR {action}: {pr_title}"
                            elif ev_type == "PushEvent":
                                n_commits = len(payload.get("commits", []))
                                title = f"Push: {n_commits} commit(s) to {payload.get('ref', '')}"

                            events.append(Event(
                                source="github",
                                kind=EventKind.NEW_CONTENT,
                                title=title,
                                detail=f"by {ev.get('actor', {}).get('login', '?')}",
                                metadata={
                                    "event_type": ev_type,
                                    "action": action,
                                    "repo": f"{self._owner}/{self._repo}",
                                },
                            ))

                if self._issue_labels:
                    resp = await client.get(
                        f"{_BASE}/repos/{self._owner}/{self._repo}/issues",
                        headers=headers,
                        params={
                            "state": "open",
                            "labels": ",".join(self._issue_labels),
                            "per_page": "20",
                            "sort": "created",
                            "direction": "desc",
                        },
                    )
                    if resp.status_code < 400:
                        issues = resp.json()
                        for iss in (issues if isinstance(issues, list) else []):
                            if "pull_request" in iss:
                                continue
                            num = iss.get("number", 0)
                            if num in self._seen_issue_numbers:
                                continue
                            self._seen_issue_numbers.add(num)

                            labels = [
                                lb.get("name", "")
                                for lb in iss.get("labels", [])
                            ]
                            events.append(Event(
                                source="github",
                                kind=EventKind.REQUEST,
                                title=f"Issue #{num}: {iss.get('title', '')}",
                                detail=_truncate(
                                    iss.get("body", "") or "", 2000,
                                ),
                                metadata={
                                    "issue_number": num,
                                    "labels": labels,
                                    "assignee": (
                                        (iss.get("assignee") or {})
                                        .get("login", "")
                                    ),
                                    "url": iss.get("html_url", ""),
                                    "repo": f"{self._owner}/{self._repo}",
                                },
                            ))

        except Exception as exc:
            logger.warning("github_perception_error", error=str(exc))

        return events

    async def mark_notifications_read(self) -> None:
        """Mark all repo notifications as read."""
        if not self._auth:
            return
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                await client.put(
                    f"{_BASE}/notifications",
                    headers=await self._auth.headers(),
                    json={"read": True},
                )
            self._seen_notification_ids.clear()
            logger.info("github_notifications_marked_read")
        except Exception as exc:
            logger.warning("github_mark_read_failed", error=str(exc))


# ---------------------------------------------------------------------------
# Dynamic registration helpers
# ---------------------------------------------------------------------------

def _register_github_tools(
    registry: ToolRegistry,
    auth: _GitHubAuth,
    scratch_dir: Path | None = None,
) -> None:
    """Register all GitHub tools."""
    tools: list[BaseTool] = [
        GitHubGetRepoTool(auth),
        GitHubReadFileTool(auth),
        GitHubListFilesTool(auth),
        GitHubCreateBranchTool(auth),
        GitHubCommitFilesTool(auth),
        GitHubCreatePRTool(auth),
        GitHubListIssuesTool(auth),
        GitHubListPRsTool(auth),
        GitHubGetPRTool(auth),
        GitHubSearchCodeTool(auth),
        GitHubCreateIssueTool(auth),
        GitHubCommentIssueTool(auth),
        GitHubGetCheckRunsTool(auth),
    ]
    if scratch_dir is not None:
        tools.append(GitHubCloneRepoTool(auth, scratch_dir))
    for tool in tools:
        registry.register(tool)
    logger.info("github_tools_registered", count=len(tools))


def bootstrap_github(
    registry: ToolRegistry,
    scratch_dir: Path | None = None,
) -> _GitHubAuth | None:
    """Bootstrap GitHub tools into the registry if credentials are available.

    Returns the auth instance (needed for perception source) or None.
    """
    config = _load_github_config()
    if not config:
        logger.debug("github_no_credentials", hint="Set GITHUB_TOKEN or GITHUB_APP_* env vars")
        return None

    auth = _GitHubAuth(config)
    _register_github_tools(registry, auth, scratch_dir=scratch_dir)
    logger.info("github_bootstrap_ok", auth_mode=config.get("auth_mode", "?"))
    return auth
