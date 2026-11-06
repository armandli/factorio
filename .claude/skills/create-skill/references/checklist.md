# Skill Validation Checklist

Run through this checklist before considering a skill complete.

## Before You Start

- [ ] 2-3 concrete use cases identified with trigger, steps, and expected result
- [ ] Required tools/MCPs identified
- [ ] Folder structure planned (SKILL.md + optional references/ and scripts/)
- [ ] Skill pattern selected (sequential, multi-MCP, iterative, context-aware, or domain-specific)

## During Development

### Structure
- [ ] Folder name is kebab-case (`my-skill`, not `my_skill` or `MySkill`)
- [ ] File is exactly `SKILL.md` (not `SKILL.MD`, `skill.md`, or `Skill.md`)
- [ ] No `README.md` inside the skill folder
- [ ] No "claude" or "anthropic" in skill name
- [ ] Frontmatter has `---` delimiters on both sides

### Description
- [ ] Describes WHAT the skill does
- [ ] Describes WHEN to use it (trigger conditions)
- [ ] Includes 2-4 natural trigger phrases
- [ ] Includes negative triggers (when NOT to use)
- [ ] Under 1024 characters
- [ ] No XML angle brackets (`< >`)
- [ ] Keyword-rich for natural language matching
- [ ] Implicit context made explicit (no assumed domain jargon)
- [ ] Distinct from other skills to avoid ambiguity

### Instructions
- [ ] Critical instructions are at the TOP, not buried
- [ ] Uses `##` headers for clear section organization
- [ ] Specific and actionable (commands, not vague directives)
- [ ] No ambiguous language ("validate properly" → explicit checks)
- [ ] Right altitude: not overly rigid if-else trees, not vague hand-waving
- [ ] Error handling is actionable (steers toward recovery, not just diagnosis)
- [ ] Diverse examples provided (2-3 canonical scenarios > exhaustive edge-case lists)
- [ ] References linked from SKILL.md with clear descriptions of what each contains

### Size
- [ ] SKILL.md body under 500 lines / 5,000 words
- [ ] Description under 1024 characters
- [ ] Each reference file under 10,000 words
- [ ] Detailed docs moved to `references/`, not inlined

### Hooks
- [ ] Evaluated whether any steps need deterministic enforcement via hooks
- [ ] If hooks included: JSON configuration provided for the correct settings file
- [ ] If hooks included: platform-specific commands provided where needed (macOS/Linux/Windows)
- [ ] If hooks included: hook scripts are marked executable (`chmod +x`)
- [ ] If hooks included: `Stop` hooks guard against infinite loops (check `stop_hook_active`)
- [ ] If hooks included: hooks tested manually with sample JSON piped to stdin

### Invocation
- [ ] Invocation mode set correctly (default / `disable-model-invocation` / `user-invocable: false`)
- [ ] `context: fork` only used for skills with explicit task instructions

## Before Shipping

### Triggering Tests
- [ ] Skill triggers on obvious matching request (e.g., "create a new component")
- [ ] Skill triggers on paraphrased request (e.g., "set up a component for me")
- [ ] Skill does NOT trigger on unrelated topics
- [ ] Skill does NOT trigger on informational questions about the topic

### Functional Tests
- [ ] Run through at least one use case end-to-end with a realistic, multi-step scenario
- [ ] Valid outputs produced
- [ ] Error handling works (intentionally trigger an error)
- [ ] Edge cases considered

### Debug Check
- [ ] Asked Claude "When would you use the [skill-name] skill?" — description is clear and complete

## After Shipping

- [ ] Tested in real conversations (not just contrived examples)
- [ ] Monitored for over-triggering (activating when it shouldn't)
- [ ] Monitored for under-triggering (not activating when it should)
- [ ] Collected user feedback
- [ ] Iterated on description and instructions based on real usage
