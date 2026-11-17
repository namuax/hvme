# Evaluation

Results are in [`../bench/`](../bench/). This page describes how they were measured.

## Benchmarks

| Set | Size | What it tests | Scoring |
|---|---|---|---|
| CaLM-lite | 45 English tasks × 30 | association, intervention, counterfactual, text and common-sense causality | final answer of the last paragraph (`humm.bench.calm`) |
| CauGym | 6 sets × 7 types × 15 | numeric causal questions in rewritten passages | exact answer |
| CLadder | 150 (50 per rung) | formal causal questions in templated stories | yes/no |
| Corr2Cause | 300 | what (in)dependence statements imply | yes/no, accuracy and F1 |
| KoCLadder | 528 | CLadder machine-translated to Korean | yes/no, with the same items in English |
| KoCorr2Cause | 1,157 | Corr2Cause machine-translated to Korean | yes/no |
| Gym v3 test | 520 | held-out generator questions | exact answer |
| KMMLU | 20 per subject (900) | Korean general knowledge, to watch for regressions | multiple choice |

The target metric for the LM is the mean accuracy of CaLM-lite and CauGym. CLadder is tracked as a regression check because the base 9B model already scores 86.7 on it.

The benchmarks are downloaded from their original sources by `humm.bench.datasets`; they are not redistributed here.

## Modes

- **tool**: the LM may call `causal_engine`, `causal_graph` and `causal_discovery` (see [tools.md](tools.md)).
- **route** (CaLM): numeric and structural tasks use tools; text and common-sense causality is answered without tools.
- **CausalCoT**: no tools, step-by-step reasoning with up to 8,192 new tokens. Used for the base models.

Decoding is greedy in bf16. Most runs used vLLM 0.33 through `humm.providers.remote.RemoteProvider`. Hugging Face and vLLM gave identical answers on the Gym test set.

## Korean sets

KoCLadder and KoCorr2Cause were translated by a base Qwen3.5-9B and kept only when automatic checks passed:

- all numbers preserved,
- enough Hangul and no leftover English function words,
- back-translation keeps the polarity of the question,
- for "Method 1 vs Method 2" questions, the comparison order is kept.

They have not been reviewed by people. On KoCLadder, only 24 of 528 items are answered wrongly in Korean by every model while English is right. That bounds how much of the Korean–English gap translation noise can explain.

## Noise

Retraining on the same data with a different seed changes CLadder 150 by up to about 8 items, tool-mode Gym by 2–3 items and KMMLU by up to 2 points. We use paired sign tests on items both models answered, and treat smaller gaps as noise.
