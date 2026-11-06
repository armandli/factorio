---
name: pull
description: Syncs the local main branch with the default remote by running git pull. Switches to main first (discarding any uncommitted changes) if not already on it. Resolves merge conflicts by reverting local changes in favor of remote. Use when user asks to "pull", "sync with remote", "update main", "pull from remote", or "sync main branch". Do NOT use for pushing changes or creating branches.
disable-model-invocation: true
---

Sync the local main branch with the default remote.

Work through these steps sequentially.

---

## Step 1 — Check Current Branch

Run `git branch --show-current`.

- If already on `main`, skip to Step 3.
- If on another branch, proceed to Step 2.

---

## Step 2 — Discard Uncommitted Changes and Switch to Main

Run `git status` to show the user what will be discarded.

Warn the user: "You have uncommitted changes on `<branch>`. They will be discarded."

Then run in sequence:

```
git restore --staged .
git restore .
git clean -fd
git checkout main
```

---

## Step 3 — Pull from Remote

Run `git pull`.

- If the pull succeeds (clean fast-forward or already up-to-date), skip to Step 5.
- If the pull fails (merge conflict or other error), proceed to Step 4.

---

## Step 4 — Resolve Conflicts by Reverting Local Changes

Run in sequence:

```
git merge --abort
git fetch origin
git reset --hard origin/main
```

Warn the user: "Local changes on main were reverted to match origin/main."

Then run `git pull` again.

If it still fails, report the error and stop.

---

## Step 5 — Report

Run `git log --oneline -5` to show recent commits.

Report:
- The current branch name
- Whether uncommitted changes were discarded (Step 2)
- Whether conflicts were resolved by reverting local changes (Step 4)
- Final sync status (up-to-date, fast-forwarded, or error)
