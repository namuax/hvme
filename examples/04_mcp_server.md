# Using humm from an MCP client

Install the MCP extra and register the server with your client. With Claude Code:

```bash
pip install "humm[mcp]"
claude mcp add humm -e HUMM_HOME=$PWD/humm_work -- humm-mcp
```

Any client that reads a JSON server list works the same way:

```json
{
  "mcpServers": {
    "humm": {
      "command": "humm-mcp",
      "env": {"HUMM_HOME": "/path/to/humm_work", "HUMM_LLM": "hf:Qwen/Qwen3.5-9B@cuda:0"}
    }
  }
}
```

`HUMM_LLM` is only needed for tools that read documents or analyse free-form questions with an LLM. The graph, validation and engine tools run without it.

Typical order of tool calls (the skills in `skills/` describe it in detail):

1. `graph_from_table` and/or `graph_from_text`, then `graph_merge`
2. `validate_run` — stop and report if the verdict is `fail`
3. `engine_identify` → `engine_estimate` → `engine_refute`
4. `engine_counterfactual` for ETT, NDE, NIE
5. or simply `ask` / `ask_v1`, which runs the whole chain and explains the answer
