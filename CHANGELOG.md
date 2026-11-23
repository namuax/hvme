# Changelog

## 0.1.1 (2026-11-23)

- README and `bench/` numbers now come from the published models: LM 9B v2.19 (CaLM-lite 91.8, CauGym 75.7, CLadder 150 90.0, KoCLadder 0.787 as the mean of three runs, KoCorr2Cause 0.976) and Router 4B ko3 seed 1 (CLadder Rung 0.997, KoCLadder Rung 0.987). 0.1.0 showed numbers from an earlier checkpoint.
- `hvme.__version__` matches the package version (0.1.0 reported 0.0.1).
- GitHub Actions moved to Node 24 releases (checkout v5, setup-python v6, upload/download-artifact v7).

## 0.1.0 (2026-11-23)

- First PyPI release. Model weights are on Hugging Face: `namuax/hvme-lm-9b`, `hvme-lm-4b`, `hvme-fast-4b`, `hvme-router-4b`.
- `FastJudge.load` reads `hvme_fast.json` (calibrated temperatures) from a Hugging Face repo id as well as from a local folder.

- First public tree: causal graph schema (HCG), graph construction from tables and text, DAG validation, symbolic causal engine, Gym question generator, single-pass judge, LM tool trajectories and training code, orchestrator, MCP server (17 tools) and nine agent skills.
- Package renamed to `hvme` (import `hvme`, CLI `hvme-mcp`, environment variables `HVME_*`).
- All code comments, docstrings, notebooks and skills are in English. Korean remains only where it is functional (Korean prompts, question templates, answer text, test inputs) and in `README.ko.md`.
- Added `docs/` (architecture, HCG format, engine, LM tools, training, evaluation), `bench/` (result summaries behind the README numbers), more `examples/` and an architecture diagram.
- `causal_discovery` reads more Korean condition phrasings ("B 를 주었을 때", "하에서", "일 때", condition after the pair). With it, the 9B LM scores the same on Korean Corr2Cause as on English (0.976).
- Added `hvme.gym.kosurf` (Gym v3.10): renders English CLadder-style bare items directly into Korean from templates (graph, probability statements, rules, query), keeping the English labels and engine values, so no translation check is needed.
- Explanations label correlation-vs-causation questions as Rung 2 (`orchestrator.explain.RUNG["association"]`), the same rule CLadder uses for backdoor-adjustment questions. Routing is unchanged.
- The engine prints values below the printed precision as 0 (`+0.0000`). Fit residue such as -2.7e-12 used to print as `-0.0000`, and the LM read it as negative.
