# Architecture

humm keeps causal knowledge in an explicit graph and does every calculation in a symbolic engine. Language models only translate: from documents to candidate edges, from a question to engine arguments, and from engine output back to an answer. Each step leaves something you can check.

```
               ┌──────────── graph construction ────────────┐
 table (CSV) ──┤ profile → PC with background knowledge     │
               │ → LLM pairwise direction → bootstrap       ├─> HCG ──> validation ──> engine ──> answer
 documents ────┤ extraction → normalisation → cross-doc     │     (L1/L2/L4)    (identify,        + paths
               │ aggregation → merge with the table graph   │                   estimate,         + assumptions
               └────────────────────────────────────────────┘                   refute, CF)       + checks
                                                                                   ▲
 question ──> router (Rung 1/2/3, single pass) ──> LM (tool calls) ───────────────┘
```

## Components

| Component | Package | What it does |
|---|---|---|
| HCG | `humm.graph.schema` | Nodes, edges with sign, lag, mechanism, evidence and status (`hypothesis`, `supported`, `refuted`, …), assumptions and provenance. See [hcg.md](hcg.md). |
| Table → HCG | `humm.graph.tabular` | PC algorithm with soft or hard background knowledge, LLM pairwise direction or causal order, multi-model agreement, bootstrap edge frequency, conflict list. |
| Text → HCG | `humm.graph.text` | Sentence splitting, causal-claim extraction, variable normalisation, cross-document aggregation, merge with the table graph by direction evidence. |
| Validation | `humm.validate` | L1 structure (cycles, time order), L2 data falsification (local Markov conditions with a permutation baseline, edge tests, mechanism fit), L4 experiment comparison, L6 benchmark range checks. |
| Engine | `humm.engine` | d-separation, identification (generalised adjustment, O-set, front-door, instrumental variables, ID algorithm), binary and linear SCMs, estimation with bootstrap intervals, refutation, causal discovery over Markov equivalence classes. See [engine.md](engine.md). |
| Gym | `humm.gym` | Question generator whose labels are computed by the engine. Used for training and evaluation; benchmark items are never used for training. |
| LM | `humm.lm` | Tool definitions and the tool loop, trajectory templates, rejection sampling, SFT and GRPO. See [tools.md](tools.md) and [training.md](training.md). |
| Fast | `humm.fast` | One forward pass that scores the options of a yes/no, choice or score question, with per-option-kind temperature calibration. Used as a router and as a cheap judge. |
| Orchestrator | `humm.orchestrator` | Question analysis → graph acquisition → validation → identification → estimation → refutation → graph sensitivity → explanation. v1 adds the Rung router and an independent LM check. |
| API and MCP | `humm.api`, `humm.mcp` | Path-based functions (graphs are stored as `data/graphs/<id>.hcg.json`) and an MCP server that exposes 17 of them as tools. |

## Questions and the Rung router

Questions are sorted into Pearl's three rungs before anything else runs.

- Rung 1 (association): conditional probabilities and correlations. Data alone answers them.
- Rung 2 (intervention): effects such as ATE and CDE. The graph must identify them first.
- Rung 3 (counterfactual): ETT, NDE, NIE, PN, PS. These need a fitted SCM.

The router is a 4B single-pass classifier. When its confidence is below the threshold (`HUMM_ROUTER_THRESHOLD`, default 0.9) the LM decides instead. Causal verbs in a Korean question ("줄이는가", "영향을 주는가") always keep the question at Rung 2, whatever the router says.

## Answers carry their checks

`api.ask` and `api.ask_v1` return more than the answer:

- the identification strategy and the adjustment set,
- the estimate with a bootstrap interval,
- refutation results (placebo treatment, random common cause, subsample),
- the validation verdict of the graph it used,
- causal paths from treatment to outcome and the assumptions the answer depends on,
- the comparison with an experiment, when one is given,
- in v1, whether an independent LM run through the engine agrees with the sign.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `HUMM_HOME` | repository root | where `data/`, `models/` and `results/` live |
| `HUMM_LLM` | — | LLM for graph construction and question analysis, e.g. `hf:Qwen/Qwen3.5-9B@cuda:0`, `anthropic:<model>`, `replay:<file>` |
| `HUMM_LM_URL`, `HUMM_LM_NAME` | — , `humm-lm` | OpenAI-compatible server (vLLM) for the LM |
| `HUMM_LM`, `HUMM_LM_DEVICE` | `models/humm-lm-9b-…`, `cuda:0` | local LM weights when no URL is set |
| `HUMM_FAST`, `HUMM_FAST_THRESHOLD`, `HUMM_FAST_DEVICE` | `models/humm-fast-4b-…`, 0.8, `cuda:0` | single-pass judge |
| `HUMM_ROUTER`, `HUMM_ROUTER_THRESHOLD` | `models/humm-fast-4b-v2-router-ko`, 0.9 | Rung router |

LLM responses can be recorded with `CachedProvider` and replayed without a GPU with `ReplayProvider`, so notebooks and tests are reproducible.
