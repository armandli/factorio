---
name: create-skill
description: Creates new Claude Code skills with optimized structure, context-efficient descriptions, and best-practice patterns. Use when user asks to "create a skill", "build a skill", "make a new skill", or "set up a skill for X". Do NOT use for general questions about skills or explaining what skills are.
argument-hint: "[skill-name] [description of what it should do]"
---

## Critical: Read This First

You are creating a Claude Code skill. A skill is a SKILL.md file (with optional references) that teaches Claude how to perform a specific repeatable task. Follow this guide precisely — small mistakes in naming, casing, or structure will cause the skill to fail silently.

## Workflow

Follow these steps in order. Do not skip steps.

### Step 1: Gather Use Cases

Before writing anything, identify 2-3 concrete use cases. For each, define:
- **Trigger**: What the user says (e.g., "deploy to staging", "create a new component")
- **Steps**: What Claude should do
- **Result**: What success looks like

Ask the user if they haven't provided this. Do not proceed without at least one clear use case.

### Step 2: Choose Approach

- **Problem-first**: User describes an outcome ("I need to set up project workspaces"). Skill orchestrates the right tool calls.
- **Tool-first**: User has MCP tools connected ("I have Notion MCP"). Skill teaches optimal workflows and best practices for those tools.

### Step 3: Choose a Pattern

Select the pattern that best fits the use cases. See [references/patterns.md](references/patterns.md) for detailed descriptions of each:

1. **Sequential Workflow Orchestration** — ordered steps with gates between them
2. **Multi-MCP Coordination** — coordinates calls across multiple MCP servers
3. **Iterative Refinement** — generate, validate, refine loop
4. **Context-Aware Tool Selection** — decision tree picks the right tool
5. **Domain-Specific Intelligence** — embeds specialized knowledge and compliance checks

### Step 4: Create the Skill

#### 4a. Directory Structure

```
skill-name/
  SKILL.md              # Required. Main instructions.
  references/           # Optional. Detailed docs loaded on demand.
    api-reference.md
    examples.md
  scripts/              # Optional. Validation/helper scripts.
    validate.py
```

#### 4b. Naming Rules

- Folder name: **kebab-case only**. `notion-project-setup`, NOT `Notion Project Setup`, `notion_project_setup`, or `NotionProjectSetup`
- File must be exactly `SKILL.md` — not `SKILL.MD`, `skill.md`, or `Skill.md`
- No `README.md` inside the skill folder
- No "claude" or "anthropic" in skill names (reserved)
- Name field: lowercase letters, numbers, hyphens only (max 64 chars)
- The folder name becomes the `/slash-command` if no `name` field is set

#### 4c. Write Frontmatter

```yaml
---
name: skill-name
description: [What it does] + [When to use it] + [Key capabilities]. Use when user asks to "[trigger phrase 1]", "[trigger phrase 2]", or "[trigger phrase 3]". Do NOT use for [negative trigger].
argument-hint: "[arg1] [arg2]"
---
```

**Description rules:**
- Under 1024 characters, no XML angle brackets (`< >`)
- Must include BOTH what the skill does AND when to use it
- Include 2-4 specific trigger phrases users would naturally say
- Add negative triggers to prevent over-firing
- Be keyword-rich and distinct from other skills
- Write as if onboarding a new team member — make implicit context explicit. Don't assume Claude knows your domain jargon or unstated conventions

**Good descriptions:**
```
description: Explains code with visual diagrams and analogies. Use when explaining how code works, teaching about a codebase, or when the user asks "how does this work?"
```
```
description: Creates weekly company newsletters following brand guidelines. Use when user asks to "write the newsletter", "draft this week's update", or "create a company update".
```

**Bad descriptions:**
- `"Helps with projects."` — too vague, no triggers
- `"Creates sophisticated multi-page documentation systems."` — no triggers
- `"Helps with newsletters"` — too generic, no specifics

#### 4d. Write Instructions (SKILL.md Body)

**Structure with `##` headers.** Put critical instructions at the top.

**Find the right altitude** — specific enough to be actionable, flexible enough to generalize:
- Too vague: `Validate the data before proceeding` / `Make sure to validate things properly`
- Too rigid: A brittle if-else tree covering every possible input combination
- Right altitude: `Before calling create_project, verify: project name non-empty, at least one team member assigned, start date not in past` — concrete examples that Claude can generalize to similar situations

**Prefer examples over edge-case lists.** A few diverse, representative examples teach Claude more than an exhaustive list of edge cases. Examples are worth a thousand words of explanation — show the expected input/output for 2-3 canonical scenarios rather than enumerating every possible variation.

**For critical validations, bundle a script** rather than relying on language instructions — code is deterministic.

**Always include explicit testing instructions.** Agents skip end-to-end verification unless told to test. Don't assume Claude will test on its own — specify exactly how to verify (e.g., "run `npm test`", "open the page in a browser via Puppeteer", "curl the endpoint and check the response"). Vague "make sure it works" instructions get skipped.

**Write actionable error handling.** Error instructions should steer Claude toward recovery, not just report failure. Instead of "if the API call fails, handle the error", write "if the API returns 429, wait 5 seconds and retry once; if it returns 401, stop and ask the user to check their API key." Guide toward the fix, not just the diagnosis.

**Two content types:**
- **Reference content** — conventions, patterns, style guides (runs inline)
- **Task content** — step-by-step actions like deploy/commit (often with `disable-model-invocation: true`)

#### 4e. Add References (if needed)

Move detailed docs, API specs, and examples to `references/`. Link them from SKILL.md:

```markdown
## Additional Resources
- For complete API details, see [references/api-reference.md](references/api-reference.md)
- For usage examples, see [references/examples.md](references/examples.md)
```

This preserves context budget. Claude loads references only when needed.

### Step 5: Consider Hooks

Evaluate whether the skill should recommend or include hook configurations. Hooks are shell commands that run automatically at specific lifecycle points — they provide deterministic guarantees that LLM instructions cannot.

**Add hooks when the skill needs:**
- **Automatic formatting** after file edits (e.g., Prettier via `PostToolUse`)
- **File protection** to block edits to sensitive files (e.g., `.env` via `PreToolUse`)
- **Notifications** when Claude needs user input (via `Notification`)
- **Context re-injection** after compaction (via `SessionStart` with `compact` matcher)
- **Completion verification** to ensure all tasks are done before stopping (via `Stop` with agent hooks)
- **Command validation** to block dangerous operations (via `PreToolUse` on `Bash`)

**Do NOT use hooks when** the decision requires LLM judgment, context varies per invocation, or the action is a one-time step within the skill's workflow.

If the skill includes hooks, provide the JSON configuration the user should add to their settings file, and include platform-specific commands where needed (macOS/Linux/Windows). See [references/hooks-best-practices.md](references/hooks-best-practices.md) for the full hooks guide, configuration format, and common patterns.

### Step 6: Configure Invocation (if needed)

| Setting | User invokes | Claude invokes | Description in context |
|---------|-------------|---------------|----------------------|
| (default) | Yes | Yes | Always loaded |
| `disable-model-invocation: true` | Yes | No | Not loaded |
| `user-invocable: false` | No | Yes | Always loaded |

- **`disable-model-invocation: true`**: For workflows with side effects — `/commit`, `/deploy`, `/send-slack-message`. Only the user can trigger it.
- **`user-invocable: false`**: For background knowledge that isn't a command — e.g., `legacy-system-context`. Hidden from `/` menu.
- **`context: fork`**: Runs in isolated subagent. No conversation history. Only for skills with explicit task instructions, not guidelines.

### Step 7: Validate

Run through the checklist at [references/checklist.md](references/checklist.md) before considering the skill complete.

## Size Limits

Keep within these bounds for optimal performance:

| Level | What | Budget |
|-------|------|--------|
| Frontmatter description | Always loaded | < 1024 chars |
| SKILL.md body | Loaded on invocation | < 500 lines / 5,000 words |
| Each reference file | Loaded on demand | < 10,000 words |

If a skill exceeds these limits, split into references or consider breaking into multiple skills.

**Context efficiency principle:** Every token in a skill depletes Claude's attention budget. Aim for the smallest set of high-signal content that produces the desired behavior. Move static reference material (API docs, schemas, style guides) into `references/` so it's loaded on demand, and keep SKILL.md focused on instructions and decision-making.

## Advanced Features

**String substitutions** available in SKILL.md:
- `$ARGUMENTS` — full argument string
- `$ARGUMENTS[0]`, `$ARGUMENTS[1]` — positional args
- `$1`, `$2` — shorthand for positional args
- `${CLAUDE_SESSION_ID}` — current session ID

**Dynamic context injection**: You can run a shell command while the skill loads and splice its stdout into the content. Prefix a single-backtick-quoted command with a bang (the exclamation-mark character); its output replaces that placeholder before the content is sent to Claude. This document deliberately avoids writing the bang-backtick form literally inline, because the skill loader would try to execute it and fail with a "Shell command permission check failed for pattern" error.

**Extended thinking**: Include the word "ultrathink" anywhere in skill content to enable extended thinking mode.

## Testing Guidance

After creating the skill, guide the user through testing:

1. **Discovery test**: Ask "What skills are available?" — skill should appear
2. **Trigger test**: Use a natural phrase matching the description — skill should activate
3. **Negative trigger test**: Ask an informational question about the topic — skill should NOT activate
4. **Functional test**: Run through a real use case end-to-end
5. **Debug trick**: Ask Claude "When would you use the [skill-name] skill?" — Claude quotes the description back, revealing gaps

**Design realistic test scenarios.** Good functional tests require multi-step reasoning, not just a single tool call. A weak test: "search for user 9182." A strong test: "Customer 9182 was charged 3x for one purchase — find all related log entries and check if other customers are affected." If your test can be completed in one trivial step, it won't catch real-world failure modes.

## Additional Resources
- For the 5 proven skill patterns with details, see [references/patterns.md](references/patterns.md)
- For the full validation checklist, see [references/checklist.md](references/checklist.md)
- For hooks best practices (when and how to include hooks in skills), see [references/hooks-best-practices.md](references/hooks-best-practices.md)
