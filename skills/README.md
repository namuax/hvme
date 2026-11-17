# Hvme-Skills

A collection of Skills that make an agent use the Hvme causal tools (hvme-mcp) in the right order and with the right principles.
Every Skill follows the same skeleton: "state assumptions → check the graph → call the tool → interpret the result → disclose limitations". Each one also carries the Humean principles as a checklist: never assert causation from correlation alone, always check for hidden confounding, and state counterfactuals only on top of explicit assumptions.

## Skill list

| Name | When to use | Main tools |
|---|---|---|
| `hvme-causal-question` | Entry point for causal questions such as "why?" or "what happens if ...?" | `classify_question`, `graph_list`, `graph_get`, `ask` |
| `hvme-graph-from-table` | Building a causal graph from a CSV or DB table | `graph_from_table`, `graph_get`, `graph_merge` |
| `hvme-graph-from-text` | Extracting causal structure from reports, papers, or logs | `graph_from_text`, `graph_merge` |
| `hvme-validate-dag` | Before using a graph for analysis | `validate_run`, `graph_get` |
| `hvme-identify-estimate` | Computing an intervention effect (ATE) | `engine_identify`, `engine_estimate`, `ask`, `lm_verify` |
| `hvme-refute` | Checking results and running sensitivity analysis | `engine_refute`, `ask`, `lm_verify` |
| `hvme-fast-judge` | Bulk yes/no, multiple-choice, or scale judgments | `fast_noul`, `fast_choice`, `fast_score` |
| `hvme-counterfactual` | "What if we had done X back then?", mediation effects | `engine_counterfactual`, `engine_identify`, `ask` |
| `hvme-explain` | Reporting the final result | `ask` (`markdown`), `graph_get` |

The default flow is `hvme-causal-question` → (`hvme-graph-from-table` / `hvme-graph-from-text`) → `hvme-validate-dag` → `hvme-identify-estimate` or `hvme-counterfactual` → `hvme-refute` → `hvme-explain`.

## Using with Claude Code

1. Install the MCP server.

```bash
pip install -e ".[mcp]"
```

2. Copy the Skills or symlink them. Use `~/.claude/skills/` for all your projects, or `<project>/.claude/skills/` for a single project.

```bash
mkdir -p ~/.claude/skills
ln -s "$PWD"/skills/hvme-* ~/.claude/skills/        # or: cp -r skills/hvme-* ~/.claude/skills/
```

3. Register the MCP server (as described in `hvme/mcp/server.py`). The server must be named `hvme` so that it matches the `hvme.<tool>` notation used in the Skill bodies.

```json
{"mcpServers": {"hvme": {"command": "python", "args": ["-m", "hvme.mcp"], "cwd": "<hvme repository>", "env": {"HVME_LLM": "anthropic:claude-sonnet-5-5"}}}}
```

Put this in the project's `.mcp.json` or register it with `claude mcp add`. Claude Code reads the `description` of each SKILL.md to choose the right Skill.

## Using with other agents

An agent other than Claude Code can use the same SKILL.md bodies as a section of its system prompt, together with the same MCP server (`python -m hvme.mcp`, stdio).
If the agent cannot select Skills on its own, always include the bodies of `hvme-causal-question` and `hvme-explain`, and add the others according to the question type.
