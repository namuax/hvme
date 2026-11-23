# Architecture

hvme keeps causal knowledge in an explicit graph and does every calculation in a symbolic engine. Language models only translate: from documents to candidate edges, from a question to engine arguments, and from engine output back to an answer. Each step leaves something you can check.

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
| HCG | `hvme.graph.schema` | Nodes, edges with sign, lag, mechanism, evidence and status (`hypothesis`, `supported`, `refuted`, …), assumptions and provenance. See [hcg.md](hcg.md). |
| Table → HCG | `hvme.graph.tabular` | PC algorithm with soft or hard background knowledge, LLM pairwise direction or causal order, multi-model agreement, bootstrap edge frequency, conflict list. |
| Text → HCG | `hvme.graph.text` | Sentence splitting, causal-claim extraction, variable normalisation, cross-document aggregation, merge with the table graph by direction evidence. |
| Validation | `hvme.validate` | L1 structure (cycles, time order), L2 data falsification (local Markov conditions with a permutation baseline, edge tests, mechanism fit), L4 experiment comparison, L6 benchmark range checks. |
| Engine | `hvme.engine` | d-separation, identification (generalised adjustment, O-set, front-door, instrumental variables, ID algorithm), binary and linear SCMs, estimation with bootstrap intervals, refutation, causal discovery over Markov equivalence classes. See [engine.md](engine.md). |
| Gym | `hvme.gym` | Question generator whose labels are computed by the engine. Used for training and evaluation; benchmark items are never used for training. |
| LM | `hvme.lm` | Tool definitions and the tool loop, trajectory templates, rejection sampling, SFT and GRPO. See [tools.md](tools.md) and [training.md](training.md). |
| Fast | `hvme.fast` | One forward pass that scores the options of a yes/no, choice or score question, with per-option-kind temperature calibration. Used as a router and as a cheap judge. |
| Orchestrator | `hvme.orchestrator` | Question analysis → graph acquisition → validation → identification → estimation → refutation → graph sensitivity → explanation. v1 adds the Rung router and an independent LM check. |
| API and MCP | `hvme.api`, `hvme.mcp` | Path-based functions (graphs are stored as `data/graphs/<id>.hcg.json`) and an MCP server that exposes 17 of them as tools. |

## Questions and the Rung router

Questions are sorted into Pearl's three rungs before anything else runs.

- Rung 1 (association): conditional probabilities and correlations. Data alone answers them.
- Rung 2 (intervention): effects such as ATE and CDE. The graph must identify them first.
- Rung 3 (counterfactual): ETT, NDE, NIE, PN, PS. These need a fitted SCM.

The router is a 4B single-pass classifier. When its confidence is below the threshold (`HVME_ROUTER_THRESHOLD`, default 0.9) the LM decides instead. Causal verbs in a Korean question ("줄이는가", "영향을 주는가") always keep the question at Rung 2, whatever the router says.

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
| `HVME_HOME` | repository root | where `data/`, `models/` and `results/` live |
| `HVME_LLM` | — | LLM for graph construction and question analysis, e.g. `hf:Qwen/Qwen3.5-9B@cuda:0`, `anthropic:<model>`, `replay:<file>` |
| `HVME_LM_URL`, `HVME_LM_NAME` | — , `hvme-lm` | OpenAI-compatible server (vLLM) for the LM |
| `HVME_LM`, `HVME_LM_DEVICE` | `namuax/hvme-lm-9b`, `cuda:0` | LM weights (Hugging Face repo id or local folder) when no URL is set; with a URL, only the tokenizer |
| `HVME_FAST`, `HVME_FAST_THRESHOLD`, `HVME_FAST_DEVICE` | `namuax/hvme-fast-4b`, 0.8, `cuda:0` | single-pass judge (repo id or local folder) |
| `HVME_ROUTER`, `HVME_ROUTER_THRESHOLD` | `namuax/hvme-router-4b`, 0.9 | Rung router (repo id or local folder; uses `HVME_FAST_DEVICE`) |

LLM responses can be recorded with `CachedProvider` and replayed without a GPU with `ReplayProvider`, so notebooks and tests are reproducible.
