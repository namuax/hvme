"""humm-mcp: exposes the Humm graph, validation, causal engine and orchestrator as MCP tools.

Run: python -m humm.mcp   (stdio). Example Claude Desktop/Code config:
  {"mcpServers": {"humm": {"command": "python", "args": ["-m", "humm.mcp"], "cwd": "<humm repo>", "env": {"HUMM_LLM": "anthropic:claude-sonnet-5-5"}}}}
Tool names follow the humm.graph.* / humm.engine.* / humm.validate.* namespaces, adapted to MCP naming rules (letters, digits, _).
"""
import json

from mcp.server.fastmcp import FastMCP

from .. import api

mcp = FastMCP("humm", instructions=(
    "Humm causal toolkit. Build a causal graph (HCG) from a CSV and/or documents, validate it against data, then answer causal questions. "
    "Typical flow: graph_from_table → (graph_from_text → graph_merge) → validate_run → ask. "
    "Every causal number must come from these tools; report the evidence path, assumptions and graph sensitivity that `ask` returns."))


def _json(x) -> str:
    return json.dumps(x, ensure_ascii=False, default=str)


@mcp.tool()
def graph_from_table(csv: str, graph_id: str, domain: str = "", alpha: float = 0.01, mode: str = "hybrid", llm: str = "") -> str:
    """Build a causal graph from a CSV file (PC discovery + optional LLM pairwise priors + data validation). Saves it under graph_id."""
    return _json(api.graph_from_table(csv, graph_id, domain, alpha, mode, llm or None))


@mcp.tool()
def graph_from_text(docs: str, graph_id: str, domain: str = "", llm: str = "") -> str:
    """Build a causal graph from documents (JSONL with id/text, or a folder of .txt/.md). Needs an LLM (llm spec or HUMM_LLM)."""
    return _json(api.graph_from_text(docs, graph_id, domain, llm or None))


@mcp.tool()
def graph_merge(text_graph: str, table_graph: str, graph_id: str, column_aliases: dict | None = None) -> str:
    """Merge a text-derived graph into a data-derived graph. column_aliases maps column → list of names used in the documents."""
    return _json(api.graph_merge(text_graph, table_graph, graph_id, column_aliases))


@mcp.tool()
def graph_get(graph_id: str) -> str:
    """Return the nodes and edges (status, confidence, evidence kinds) of a saved graph."""
    return _json(api.graph_get(graph_id))


@mcp.tool()
def graph_list() -> str:
    """List saved graph ids."""
    return _json(api.graph_list())


@mcp.tool()
def validate_run(graph_id: str, csv: str, n_perm: int = 100) -> str:
    """Validate a graph against data (L1 structure, L2 conditional-independence falsification). Updates edge statuses."""
    return _json(api.validate_run(graph_id, csv, n_perm))


@mcp.tool()
def engine_identify(graph_id: str, treatment: str, outcome: str, csv: str = "") -> str:
    """Identify the causal effect of treatment on outcome (backdoor, frontdoor, IV). Columns missing from csv are treated as latent."""
    return _json(api.engine_identify(graph_id, treatment, outcome, csv or None))


@mcp.tool()
def engine_estimate(graph_id: str, csv: str, treatment: str, outcome: str, n_boot: int = 200) -> str:
    """Estimate the identified average causal effect with a bootstrap 95% interval."""
    return _json(api.engine_estimate(graph_id, csv, treatment, outcome, n_boot))


@mcp.tool()
def engine_refute(graph_id: str, csv: str, treatment: str, outcome: str) -> str:
    """Run refutation tests (placebo treatment, random common cause, data subset) on the estimate."""
    return _json(api.engine_refute(graph_id, csv, treatment, outcome))


@mcp.tool()
def engine_counterfactual(graph_id: str, csv: str, treatment: str, outcome: str, query: str = "ett", mediators: list[str] | None = None) -> str:
    """Counterfactual quantity from a binary SCM fitted to the data: ate, ett, nde, nie, pn, ps."""
    return _json(api.engine_counterfactual(graph_id, csv, treatment, outcome, query, mediators))


@mcp.tool()
def ask(question: str, graph_id: str, csv: str, experiment_csv: str = "", experiment_treatment: str = "", llm: str = "") -> str:
    """Answer a causal question end to end: parse → validate → identify → estimate → refute → graph sensitivity → explanation (markdown)."""
    return _json(api.ask(question, graph_id, csv, experiment_csv or None, experiment_treatment or None, llm or None))


@mcp.tool()
def fast_noul(proposition: str, context: str = "") -> str:
    """Humm-Fast single-pass yes/no judgement. If escalate is true, do not use p_yes; route to ask or lm_verify instead."""
    return _json(api.fast_noul(proposition, context))


@mcp.tool()
def fast_choice(question: str, options: list[str], context: str = "") -> str:
    """Humm-Fast single-pass multiple choice. If escalate is true, the answer is low-confidence and must be escalated."""
    return _json(api.fast_choice(question, options, context))


@mcp.tool()
def fast_score(question: str, levels: int = 5, context: str = "") -> str:
    """Humm-Fast single-pass ordinal score (1..levels) with its distribution. Escalate when escalate is true."""
    return _json(api.fast_score(question, levels, context))


@mcp.tool()
def classify_question(question: str, graph_id: str = "") -> str:
    """Classify a causal question by rung (1 association, 2 intervention, 3 counterfactual) with Humm-Fast, escalating to Humm-LM
    when unsure. With graph_id, also map treatment/outcome/mediators onto graph variables."""
    return _json(api.classify_question(question, graph_id))


@mcp.tool()
def lm_verify(graph_id: str, csv: str, treatment: str, outcome: str) -> str:
    """Humm-LM (tool mode) independently answers whether the treatment raises the outcome from the verbalized graph and fitted
    probabilities; agree compares it with the sign of the engine ATE (None when the engine CI contains 0)."""
    return _json({k: v for k, v in api.lm_verify(graph_id, csv, treatment, outcome).items() if k != "trace"})


@mcp.tool()
def ask_v1(question: str, graph_id: str, csv: str, experiment_csv: str = "", experiment_treatment: str = "", verify: bool = True) -> str:
    """Like ask, plus orchestrator v1 steps: Humm-Fast rung routing for the question kind and a Humm-LM independent check."""
    return _json(api.ask_v1(question, graph_id, csv, experiment_csv or None, experiment_treatment or None, verify=verify))


def main() -> None:
    mcp.run()
