# Benchmark results

The numbers in the top-level README come from these files. Per-item predictions are not included because the benchmark items themselves are not redistributed.

| File | Model | Contents |
|---|---|---|
| `lm_9b.json` | humm LM 9B, tool mode | CaLM-lite (by task), CauGym (by set and type), CLadder 150 (by query type), Corr2Cause 300 (accuracy, F1), KoCLadder 528 and KoCorr2Cause 1,157 (Korean and the same items in English), KMMLU 900, Gym v3 520 |
| `base_9b.json` | Qwen3.5-9B, no tools | CaLM-lite, CauGym, CLadder 150 with CausalCoT (8k tokens) |
| `fast_4b.json` | humm Fast 4B | single-pass accuracy and expected calibration error on CLadder 600, Corr2Cause, Gym v3 test, CauGym yes/no, CaLM `-B` |
| `router_4b.json` | humm Router 4B | Rung accuracy on CLadder 600, KoCLadder and 18 Korean demo questions |

Settings: greedy decoding, bf16, served with vLLM 0.33. CaLM-lite is 45 English tasks × 30 items and is scored on the final answer only. CauGym is 6 sets × 7 types × 15 items. The Korean sets are machine translations that passed automatic checks and have not been reviewed by people.
