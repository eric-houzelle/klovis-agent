---
name: github
version: "1.0"
description: GitHub REST API for repository management, pull requests, issues, and code search.
homepage: https://github.com
api_base: https://api.github.com
auth: bearer
auth_env: GITHUB_TOKEN
---

# GitHub API

REST API for managing repositories, branches, commits, pull requests, and issues.

**Base URL:** `https://api.github.com`

**Authentication:** All requests require a `Bearer` token via the `Authorization` header.
The token can be a Personal Access Token (PAT) or a GitHub App installation token.

## Repositories

### Get repository info
```
GET /repos/{owner}/{repo}
```
Returns metadata: default branch, description, language, open issues count, etc.

## Contents

### Read a file
```
GET /repos/{owner}/{repo}/contents/{path}?ref={branch}
```
Returns base64-encoded content. Decode with `base64.b64decode(data["content"])`.

### List directory contents
```
GET /repos/{owner}/{repo}/contents/{path}?ref={branch}
```
When `path` is a directory, returns an array of entries with `name`, `type`, `path`, `size`.

## Branches

### Create a branch
1. Get the SHA of the base ref:
```
GET /repos/{owner}/{repo}/git/ref/heads/{base_branch}
```
2. Create the new ref:
```
POST /repos/{owner}/{repo}/git/refs
{"ref": "refs/heads/{new_branch}", "sha": "{base_sha}"}
```

## Commits (via Git Trees API)

To commit files without a local clone:

1. Get the current commit SHA of the branch:
```
GET /repos/{owner}/{repo}/git/ref/heads/{branch}
```

2. Get the tree SHA of that commit:
```
GET /repos/{owner}/{repo}/git/commits/{commit_sha}
```

3. Create a new tree with file changes:
```
POST /repos/{owner}/{repo}/git/trees
{
  "base_tree": "{tree_sha}",
  "tree": [
    {"path": "file.py", "mode": "100644", "type": "blob", "content": "new content"}
  ]
}
```

4. Create a new commit:
```
POST /repos/{owner}/{repo}/git/commits
{"message": "commit msg", "tree": "{new_tree_sha}", "parents": ["{parent_sha}"]}
```

5. Update the branch ref:
```
PATCH /repos/{owner}/{repo}/git/refs/heads/{branch}
{"sha": "{new_commit_sha}"}
```

## Pull Requests

### Create a PR
```
POST /repos/{owner}/{repo}/pulls
{
  "title": "PR title",
  "body": "Description in markdown",
  "head": "feature-branch",
  "base": "main",
  "draft": false
}
```

### List PRs
```
GET /repos/{owner}/{repo}/pulls?state=open&sort=created&direction=desc
```

### Get PR details
```
GET /repos/{owner}/{repo}/pulls/{pr_number}
```

### Get PR diff
```
GET /repos/{owner}/{repo}/pulls/{pr_number}
Accept: application/vnd.github.diff
```

### Get PR reviews
```
GET /repos/{owner}/{repo}/pulls/{pr_number}/reviews
```

## Issues

### List issues
```
GET /repos/{owner}/{repo}/issues?state=open&labels=bug&sort=updated
```

### Get a single issue
```
GET /repos/{owner}/{repo}/issues/{issue_number}
```

## Code Search

```
GET /search/code?q={query}+repo:{owner}/{repo}
```

Search syntax supports: `path:`, `extension:`, `language:`, `filename:`.

Example: `GET /search/code?q=def+bootstrap+repo:user/repo+path:src/`

## Rate Limits

- Authenticated requests: 5,000/hour (PAT) or 5,000/hour per installation (App)
- Search API: 30 requests/minute
- Check limits: `GET /rate_limit`

## Common Headers

All requests should include:
```
Authorization: Bearer {token}
Accept: application/vnd.github+json
X-GitHub-Api-Version: 2022-11-28
```
