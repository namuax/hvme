# Changelog

## 0.1.0 (unreleased)

- First public tree: causal graph schema (HCG), graph construction from tables and text, DAG validation, symbolic causal engine, Gym question generator, single-pass judge, LM tool trajectories and training code, orchestrator, MCP server (17 tools) and nine agent skills.
- Package renamed to `hvme` (import `hvme`, CLI `hvme-mcp`, environment variables `HVME_*`).
- All code comments, docstrings, notebooks and skills are in English. Korean remains only where it is functional (Korean prompts, question templates, answer text, test inputs) and in `README.ko.md`.
- Added `docs/` (architecture, HCG format, engine, LM tools, training, evaluation), `bench/` (result summaries behind the README numbers), more `examples/` and an architecture diagram.
- `causal_discovery` reads more Korean condition phrasings ("B 를 주었을 때", "하에서", "일 때", condition after the pair). With it, the 9B LM scores the same on Korean Corr2Cause as on English (0.976).
- Added `hvme.gym.kosurf` (Gym v3.10): renders English CLadder-style bare items directly into Korean from templates (graph, probability statements, rules, query), keeping the English labels and engine values, so no translation check is needed.
- Explanations label correlation-vs-causation questions as Rung 2 (`orchestrator.explain.RUNG["association"]`), the same rule CLadder uses for backdoor-adjustment questions. Routing is unchanged.
