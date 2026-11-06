---
name: commit-push
description: Commits all current changes and pushes to remote origin on the current branch. Use when user asks to "commit and push", "commit push", or "push my changes". Do NOT use for committing without pushing or for PR creation.
disable-model-invocation: true
---

Commit all current changes and push to the remote origin on the current branch.

Work through these steps sequentially. Stop early if there are no changes.

---

## Step 1 — Check Status

Run these commands in parallel:
- `git status` (never use `-uall`)
- `git diff` to see unstaged changes
- `git diff --staged` to see already-staged changes

If there are no changes (no untracked files, no modifications, no staged changes), report "No changes to commit." and **stop**.

---

## Step 2 — Stage All Changes

Run `git add -A` to stage everything.

---

## Step 3 — Generate Commit Message

Run these commands in parallel:
- `git diff --cached` to see the full staged diff
- `git log --oneline -10` to see recent commit style

Analyze the staged diff and draft a concise commit message (1-2 sentences) that:
- Summarizes the nature of the changes (new feature, bug fix, refactor, etc.)
- Focuses on the "why" rather than the "what"
- Follows the style of recent commits in the repo

---

## Step 4 — Commit

Create the commit using a HEREDOC to ensure correct formatting:

```
git commit -m "$(cat <<'EOF'
<commit message here>

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

If the commit fails due to a pre-commit hook, fix the issue, re-stage with `git add -A`, and create a **new** commit (never amend).

---

## Step 5 — Push

Get the current branch name:

```
git branch --show-current
```

Then push to remote:

```
git push origin <branch>
```

If the branch has no upstream, use `git push -u origin <branch>` instead.

---

## Step 6 — Report

Show the user:
- The commit hash and message (`git log --oneline -1`)
- The push result (success or failure)
- The branch that was pushed
