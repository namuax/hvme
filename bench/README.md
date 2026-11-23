# Benchmark results

Models: LM 9B v2.19 (`namuax/hvme-lm-9b`), Fast 4B v2-story-distill (`namuax/hvme-fast-4b`), Router 4B ko3 seed 1 (`namuax/hvme-router-4b`).

The numbers in the top-level README come from these files. Per-item predictions are not included because the benchmark items themselves are not redistributed.

| File | Model | Contents |
|---|---|---|
| `lm_9b.json` | hvme LM 9B, tool mode | CaLM-lite (by task), CauGym (by set and type), CLadder 150 (by query type), Corr2Cause 300 (accuracy, F1), KoCLadder 528 and KoCorr2Cause 1,157 (Korean and the same items in English, mean of three vLLM runs), KMMLU 900, Gym v3 520 |
| `base_9b.json` | Qwen3.5-9B, no tools | CaLM-lite, CauGym, CLadder 150 with CausalCoT (8k tokens) |
| `fast_4b.json` | hvme Fast 4B | single-pass accuracy and expected calibration error on CLadder 600, Corr2Cause, Gym v3 test, CauGym yes/no, CaLM `-B` |
| `router_4b.json` | hvme Router 4B | Rung accuracy on CLadder 600, KoCLadder and 18 Korean demo questions; yes/no head on CLadder 600, Corr2Cause, CauGym yes/no, CaLM `-B` |

Settings: greedy decoding, bf16, served with vLLM 0.33. CaLM-lite is 45 English tasks × 30 items and is scored on the final answer only. CauGym is 6 sets × 7 types × 15 items. The Korean sets are machine translations that passed automatic checks and have not been reviewed by people.
