# Proven Skill Patterns

These are the 5 proven patterns for structuring Claude Code skills, derived from Anthropic's official guide. Choose the one that best fits your use case.

## 1. Sequential Workflow Orchestration

**When to use:** The task has ordered steps where each depends on the previous one completing successfully.

**Structure:**
- Explicit step ordering with numbered phases
- Validation gates between steps — do not proceed until the gate passes
- Rollback instructions if a step fails
- Clear dependencies between steps

**Example use cases:**
- Project setup (create repo → configure CI → set up environments → deploy)
- Release workflow (run tests → bump version → build → publish → tag)
- Database migration (backup → validate schema → migrate → verify → clean up)

**Template:**
```markdown
## Step 1: [Action]
[Instructions]

### Gate: Verify [condition] before proceeding.

## Step 2: [Action]
[Instructions]

### Gate: Verify [condition] before proceeding.

## Rollback
If any step fails: [rollback instructions]
```

**Tips:**
- **State persistence via files:** For long or resumable workflows, write intermediate results to files (e.g., `_state/step2_output.json`) between steps. This lets Claude resume from the last successful step if a workflow is interrupted. See also [Long-Running Workflow Management](#long-running-workflow-management) for multi-session techniques.

## 2. Multi-MCP Coordination

**When to use:** The task requires coordinating calls across multiple MCP servers (e.g., Notion + GitHub + Slack).

**Structure:**
- Clear phase separation — which MCP is used in each phase
- Data passing between MCPs — what output from MCP-A feeds into MCP-B
- Validation before each phase transition
- Error handling per MCP (one failing shouldn't crash the whole workflow)

**Example use cases:**
- Project kickoff (create Notion workspace → create GitHub repo → post Slack announcement)
- Incident response (query monitoring MCP → create Jira ticket → notify Slack → update status page)
- Content publishing (draft in CMS → generate assets → deploy → announce)

**Template:**
```markdown
## Phase 1: [MCP-A Action]
Using [MCP-A], do [action]. Save [output] for Phase 2.

### Validate: Confirm [MCP-A result] before proceeding.

## Phase 2: [MCP-B Action]
Using [MCP-B], do [action] with [output from Phase 1].

### Validate: Confirm [MCP-B result] before proceeding.

## Error Handling
- If [MCP-A] fails: [recovery]
- If [MCP-B] fails: [recovery]
```

**Tips:**
- **Progressive tool discovery:** When coordinating many MCP tools (dozens+), loading all tool definitions upfront can consume 150K+ tokens of context. Instead, instruct Claude to call a `list_tools` or `search_tools` method first and only load the specific tools needed for each phase. This keeps context lean and avoids hitting limits.
- **In-environment data processing:** When an MCP tool returns a large dataset (e.g., 10,000 spreadsheet rows), instruct Claude to filter and transform the data in code before passing results to the next phase. This prevents context bloat from raw data flowing through multiple tool calls.
- **Prefer consolidated tools over granular ones:** When instructing Claude to use MCP tools, favor higher-level operations over many small calls. For example, prefer `schedule_event` (which checks availability internally) over separate `list_users` → `list_events` → `create_event` sequences. Fewer tool calls means less context consumed and fewer failure points.
- **Use semantic identifiers between phases:** When passing data from one MCP to another, reference items by meaningful names (e.g., "the Acme onboarding project") rather than opaque UUIDs. This reduces hallucination risk and makes the workflow readable.

## 3. Iterative Refinement

**When to use:** The task produces output that needs quality checking and revision (creative work, code generation, documentation).

**Structure:**
- Generate initial draft
- Run quality check (ideally via a validation script, not just prose instructions)
- Refinement loop with explicit exit criteria
- Finalization step

**Example use cases:**
- Code generation (generate → lint → fix → test → finalize)
- Documentation (draft → review against style guide → revise → approve)
- Design assets (generate → check brand compliance → refine → export)

**Template:**
```markdown
## Step 1: Generate Draft
[Generation instructions]

## Step 2: Validate
Run `scripts/validate.sh` (or check against [criteria]).

## Step 3: Refine
If validation fails, fix [specific issues] and return to Step 2.
Maximum 3 refinement cycles. If still failing after 3 cycles, report issues to user.

## Step 4: Finalize
[Finalization instructions]
```

## 4. Context-Aware Tool Selection

**When to use:** The right action depends on the situation — file type, size, environment, user role, etc.

**Structure:**
- Decision tree based on context variables
- Clear branching logic
- Each branch has its own instructions
- Fallback for unrecognized contexts

**Example use cases:**
- File storage (small files → local, large files → S3, sensitive files → encrypted vault)
- Deployment (dev → local docker, staging → preview URL, production → full deploy pipeline)
- Communication (urgent → Slack DM, normal → email, FYI → Slack channel)

**Template:**
```markdown
## Determine Context
Check [variable]. Branch based on result:

### If [condition A]:
[Instructions for path A]

### If [condition B]:
[Instructions for path B]

### Otherwise:
[Fallback instructions]
```

## 5. Domain-Specific Intelligence

**When to use:** The task requires specialized knowledge that prevents common mistakes — compliance rules, security practices, architectural conventions.

**Structure:**
- Compliance/validation checks BEFORE any action
- Domain rules embedded as explicit constraints
- Audit trail — log what was checked and why
- Domain-specific error messages that explain the "why"

**Example use cases:**
- Security-hardened deployments (check OWASP top 10 before deploying)
- Financial compliance (validate regulatory requirements before processing)
- Infrastructure provisioning (enforce tagging, naming, and access policies)
- Senior engineer knowledge transfer (capture domain gotchas that prevent production failures)

**Template:**
```markdown
## Pre-Flight Checks
CRITICAL: Before proceeding, verify:
- [ ] [Domain rule 1]
- [ ] [Domain rule 2]
- [ ] [Domain rule 3]

If any check fails, STOP and report the violation. Do not proceed.

## Action
[Instructions with domain constraints embedded]

## Audit
Log the following: [what was checked, what passed, what was done]
```

## Cross-Cutting Techniques

These techniques can be layered onto any pattern above.

### Sub-Agent Delegation

When a skill involves complex research or parallel exploration, instruct Claude to delegate sub-tasks to focused agents (using the Task tool). Each sub-agent works with a clean context window, explores extensively (tens of thousands of tokens), and returns a condensed summary (1,000–2,000 tokens). The main agent then synthesizes results without being overwhelmed by raw exploration data.

**When to use:** Research-heavy steps, parallel information gathering, or any step where exploration would bloat the main conversation.

**Example:** A competitive analysis skill delegates "research competitor X" to three parallel sub-agents, each returning a structured summary, then synthesizes findings in the main context.

### Long-Running Workflow Management

For skills that may span multiple context windows, think of each context window as an engineer working a shift with no memory of previous shifts. Design for clean handoffs:

- **Structured handoff artifacts:** Maintain a progress file (e.g., `claude-progress.txt`) with completed work and next steps, and a feature/task list in JSON format — models are less likely to inappropriately modify JSON than Markdown. Use descriptive git commits as a structured change record.
- **Session startup ritual:** Begin each session by reading progress files and git log, identifying the next incomplete step, and running a sanity check before starting new work. This costs tokens upfront but prevents expensive recovery from broken state.
- **One task per session:** Complete exactly one step or feature per session and leave the environment in a clean, resumable state. Attempting too much risks running out of context mid-implementation.

**When to use:** Skills with 10+ steps, multi-session workflows, or any task where context compaction might discard important earlier decisions.

## Combining Patterns

Skills often combine multiple patterns. For example:
- A deployment skill might use **Sequential Workflow** + **Domain-Specific Intelligence** (ordered steps with security checks at each gate)
- A content pipeline might use **Multi-MCP Coordination** + **Iterative Refinement** (coordinate across tools with quality loops)

Choose the primary pattern, then layer in elements from others as needed.
