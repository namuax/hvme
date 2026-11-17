# Training recipes

All training data comes from the Gym generators (`humm.gym`), so every label is computed by the engine. Benchmark items (CLadder, Corr2Cause, CaLM, CauGym, KoCLadder, KoCorr2Cause) are only used for evaluation.

## LM (tool use)

Base: Qwen3.5-9B or Qwen3.5-4B, chat template with `enable_thinking=False`.

| Setting | Value |
|---|---|
| Method | LoRA SFT (`humm.fast.lora.add_lora`, `humm.lm.sft.train_sft`), loss on the assistant tokens only |
| LoRA | r 16, α 32 |
| Optimiser | AdamW, lr 1e-4 from the base model, 5e-5 when continuing from a fine-tuned model, 3% warm-up |
| Batch | 2 × gradient accumulation 16, examples of similar length batched together |
| Length | up to 4,096 tokens |
| Epochs | 1 |

Data, in the proportions that worked best:

1. **Tool trajectories from templates** (`humm.lm.trajectories`): for each Gym question, a trajectory that states the variables, the query type and the estimand, calls the right tool, and concludes from the engine value. Covers numeric queries (`causal_engine`), structural queries (`causal_graph`) and equivalence-class questions (`causal_discovery`).
2. **Recovery trajectories**: the first call contains a typical mistake (undeclared state, wrong tool, missing statement), the engine returns its error, and the next call fixes it. The erroneous turn is excluded from the loss.
3. **Rejection-sampled trajectories** (`humm.lm.sample`): a teacher model solves each question several times with the tools; only trajectories whose answer is correct and whose last engine value matches the Gym value are kept. This is how the Korean data was built: Gym questions were machine-translated, then solved by the English-trained model.
4. **Replay of general ability**: the base model's own answers to KMMLU training questions (self-distillation), so general knowledge does not drift.

Lessons that shaped the recipe:

- Do not put short hand-written solutions without tools in the same surface form as the benchmark; the model starts answering every question in that short form.
- Change one data source at a time and judge it on the question types it targets. Overall scores move by up to 8 items on CLadder 150 and 2 points on KMMLU between seeds.
- Many apparent model errors were engine or harness errors (state spelling, `not X`, repeated calls). Fix those in the engine first and re-evaluate before training again.

GRPO (`humm.lm.grpo`) with an exact-answer reward times a process score is implemented. In our runs it did not beat SFT on the overall metrics, so the published models are SFT models.

## Fast (single pass)

Base: Qwen3.5-4B. One forward pass scores the option tokens of a yes/no, choice or score question.

| Setting | Value |
|---|---|
| Method | LoRA, cross-entropy on the option distribution (`humm.fast.train`) |
| LoRA | r 16, α 32 |
| Optimiser | lr 1e-4, batch 8 × gradient accumulation 4, 1 epoch |
| Calibration | one temperature per option kind, fitted on a dev set (`humm.fast.calibrate`) |

The training targets are the answer distributions of the LM on Gym questions rewritten as short stories in other domains. Seed-to-seed differences on external sets reach 2–8 points, so compare Fast variants with at least three seeds.

The Rung router is the same model further trained on Rung classification (English Gym questions plus Korean free-form questions and translated Gym passages), with the original training items replayed so the yes/no judgement does not degrade.
