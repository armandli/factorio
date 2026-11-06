# Hooks Best Practices for Skill Creators

When building a skill, consider whether parts of the workflow should be enforced via hooks rather than (or in addition to) LLM instructions. Hooks provide deterministic guarantees that instructions alone cannot.

## Hooks vs Skills: When to Use Which

| Use a **skill** when... | Use a **hook** when... |
|---|---|
| The task requires LLM judgment | The action must happen every time, no exceptions |
| Steps vary based on context | The rule is simple and deterministic |
| The user triggers it explicitly | The behavior should run automatically in the background |
| Output quality matters more than consistency | Consistency and enforcement matter most |

**Use both together** when a skill defines a workflow but certain steps within it need deterministic enforcement (e.g., a deploy skill that also installs a `PreToolUse` hook to block production writes without approval).

## Hook Types

- **`command`** — Runs a shell command. Use for formatting, validation, notifications, logging.
- **`prompt`** — Sends context to a Claude model for a yes/no judgment. Use when the decision requires understanding intent, not just pattern matching.
- **`agent`** — Spawns a subagent that can read files and run tools before deciding. Use when verification requires inspecting actual codebase state.

## Hook Events Reference

| Event | When it fires | Common use |
|---|---|---|
| `SessionStart` | Session begins, resumes, or after compaction | Inject context, set up environment |
| `UserPromptSubmit` | User submits a prompt | Validate or transform user input |
| `PreToolUse` | Before a tool executes (can block) | Guard protected files, validate commands |
| `PostToolUse` | After a tool succeeds | Auto-format, log changes, trigger builds |
| `PostToolUseFailure` | After a tool fails | Custom error recovery |
| `Notification` | Claude needs user attention | Desktop notifications |
| `Stop` | Claude finishes responding | Verify task completion |
| `PermissionRequest` | Permission dialog appears | Auto-approve known-safe patterns |
| `SubagentStart` / `SubagentStop` | Subagent lifecycle | Monitor or constrain subagent behavior |
| `PreCompact` | Before context compaction | Save critical state |
| `SessionEnd` | Session terminates | Clean up temp files |

## Platform-Specific Commands

When a skill includes hook configurations that produce notifications or use OS features, provide commands for all platforms:

- **macOS**: `osascript -e 'display notification "message" with title "Title"'`
- **Linux**: `notify-send 'Title' 'message'`
- **Windows**: `powershell.exe -Command "[System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms'); [System.Windows.Forms.MessageBox]::Show('message', 'Title')"`

## Configuration Format

Hooks live in settings JSON files, not in SKILL.md. A skill can recommend or generate hook configurations for the user to add.

```json
{
  "hooks": {
    "EventName": [
      {
        "matcher": "regex_pattern",
        "hooks": [
          {
            "type": "command",
            "command": "your-shell-command"
          }
        ]
      }
    ]
  }
}
```

### Hook Storage Locations

| Location | Scope | Shareable |
|---|---|---|
| `~/.claude/settings.json` | All projects (user-level) | No |
| `.claude/settings.json` | Single project (committable) | Yes |
| `.claude/settings.local.json` | Single project (gitignored) | No |

### Matchers

Matchers are regex patterns that filter when a hook fires. Each event type matches on a specific field:

- `PreToolUse` / `PostToolUse` / `PermissionRequest`: matches **tool name** (e.g., `Bash`, `Edit|Write`, `mcp__github__.*`)
- `SessionStart`: matches **source** (`startup`, `resume`, `clear`, `compact`)
- `SessionEnd`: matches **reason** (`clear`, `logout`, `other`)
- `Notification`: matches **type** (`permission_prompt`, `idle_prompt`)

An empty or omitted matcher fires on every occurrence of that event.

## Communication Protocol

Hooks communicate with Claude Code through stdin/stdout/stderr and exit codes:

- **stdin**: Receives JSON with event data (`session_id`, `cwd`, `tool_name`, `tool_input`, etc.)
- **stdout**: Output added to Claude's context, or structured JSON for decisions
- **stderr**: Feedback message when blocking (exit 2), or debug logging
- **Exit 0**: Allow the action to proceed
- **Exit 2**: Block the action (stderr message is sent to Claude as feedback)
- **Other exit codes**: Action proceeds, stderr logged but not shown to Claude

## Best Practices When Skills Include Hooks

1. **Separate concerns**: Put hook scripts in `scripts/` or `.claude/hooks/` — not inline in settings JSON for anything beyond one-liners.

2. **Make scripts executable**: Always include `chmod +x` instructions for macOS/Linux. Hook scripts that aren't executable fail silently.

3. **Use `jq` for JSON parsing**: Hook input arrives as JSON on stdin. Parse with `jq` for reliability:
   ```bash
   INPUT=$(cat)
   FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
   ```

4. **Guard against infinite loops in Stop hooks**: If a `Stop` hook can cause Claude to continue working, check the `stop_hook_active` field to prevent infinite loops:
   ```bash
   INPUT=$(cat)
   if [ "$(echo "$INPUT" | jq -r '.stop_hook_active')" = "true" ]; then
     exit 0
   fi
   ```

5. **Keep hook commands fast**: Hooks block Claude's execution. Avoid network calls or slow operations in `PreToolUse` hooks especially. Default timeout is 10 minutes, but aim for sub-second execution.

6. **Use structured JSON output for nuanced decisions**: Instead of just allow/block, return JSON for `PreToolUse` hooks:
   ```json
   {
     "hookSpecificOutput": {
       "hookEventName": "PreToolUse",
       "permissionDecision": "deny",
       "permissionDecisionReason": "Use rg instead of grep for better performance"
     }
   }
   ```

7. **Test hooks manually before shipping**: Pipe sample JSON to verify behavior:
   ```bash
   echo '{"tool_name":"Bash","tool_input":{"command":"ls"}}' | ./my-hook.sh
   echo $?
   ```

8. **Avoid echo in shell profiles**: Hooks run in non-interactive shells that source `~/.zshrc` or `~/.bashrc`. Unconditional `echo` statements in profiles corrupt hook JSON output. Wrap them:
   ```bash
   if [[ $- == *i* ]]; then
     echo "Shell ready"
   fi
   ```

## Common Patterns for Skills That Include Hooks

### Auto-format after edits
```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "jq -r '.tool_input.file_path' | xargs npx prettier --write"
          }
        ]
      }
    ]
  }
}
```

### Block edits to protected files
```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/protect-files.sh"
          }
        ]
      }
    ]
  }
}
```

### Re-inject context after compaction
```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "compact",
        "hooks": [
          {
            "type": "command",
            "command": "echo 'Reminder: use Bun, not npm. Run bun test before committing.'"
          }
        ]
      }
    ]
  }
}
```

### Verify completion with an agent hook
```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "agent",
            "prompt": "Verify that all unit tests pass. Run the test suite and check the results.",
            "timeout": 120
          }
        ]
      }
    ]
  }
}
```
