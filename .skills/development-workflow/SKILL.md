---
name: development-workflow
version: "1.0"
description: How to develop, test, and ship code changes on a Git repository.
---

# Development Workflow

This skill describes how to work on a codebase — reading, modifying, testing,
and shipping code via Git and GitHub.

## Principle

Work locally like a developer.  Never push code you haven't tested.

## Workflow

### 1. Clone the repository

Use the **`github_clone_repo`** tool.  It handles authentication automatically
and clones the repo into your scratch workspace.  Call it with `owner` and
`repo` (and optionally `branch`).  It returns the local path where the repo
was cloned — use that path for all subsequent operations.

After cloning, all subsequent file operations happen on the local clone.

### 2. Understand the codebase

Use `file_read` and `shell_command` (e.g. `find`, `head`, `wc -l`) to explore
the project structure.  Read the README, the entry points, and the modules
relevant to your task.  Do NOT rely on `github_read_file` for this — the
local clone is faster and has no API rate limits.

### 3. Create a branch

```
cd /path/to/workspace/<repo>
git checkout -b <branch-name>
```

### 4. Make changes

Use `file_write` and `file_edit` to modify files in the local clone.  You can
create new files, edit existing ones, or delete them.

For large changes, work incrementally: modify one module, test it, then move
to the next.

### 5. Test

Before committing, always run the project's test suite and linter:

```
cd /path/to/workspace/<repo>
# Run tests (adapt to the project's tooling)
python -m pytest tests/ -x -q
# Run linter
ruff check .
# Or whatever the project uses (mypy, flake8, eslint, cargo test, etc.)
```

If tests fail, read the error output, fix the code, and re-run.  Iterate
until all tests pass and the linter is clean.

If the project has no tests yet and your change is non-trivial, consider
writing a small test to validate your work before committing.

### 6. Commit and push

```
cd /path/to/workspace/<repo>
git add -A
git commit -m "descriptive commit message"
git push origin <branch-name>
```

If the push requires authentication, use the token-injected remote URL or
set the remote explicitly:

```
git remote set-url origin https://x-access-token:<TOKEN>@github.com/<owner>/<repo>.git
```

### 7. Open a pull request

Use `github_create_pr` to open the PR.  Write a clear title and body
explaining what changed and why.

### 8. Monitor CI

After opening the PR, use `github_get_check_runs` to verify that CI passes.
If it fails, read the logs, fix locally, commit, and push again.

## When to use GitHub API tools vs local tools

| Task | Use |
|------|-----|
| Clone a repo | `github_clone_repo` |
| Read/edit files | `file_read`, `file_write`, `file_edit` |
| Run tests, linters | `shell_command` |
| Git operations (branch, commit, push) | `shell_command` |
| Open a PR | `github_create_pr` |
| Create an issue | `github_create_issue` |
| Comment on an issue/PR | `github_comment_issue` |
| Check CI status | `github_get_check_runs` |
| Browse a repo you haven't cloned | `github_read_file`, `github_list_files` |

The GitHub API tools (`github_read_file`, `github_commit_files`, etc.) are
useful for quick, lightweight operations on remote repos you haven't cloned.
But for any task that involves writing or modifying code, always prefer the
local workflow: clone → edit → test → push.

## Common pitfalls

- **Do not push untested code.**  Always run the test suite first.
- **Do not use `github_commit_files` for development.**  It bypasses local
  testing entirely.  Use it only for trivial single-file fixes (typos, docs).
- **Do not guess file paths.**  Explore the repo structure first.
- **Do not assume the repo layout.**  Read the actual directory tree, don't
  assume `src/` or `lib/` exists.
