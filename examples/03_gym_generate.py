"""Generate causal questions whose answers are computed by the engine (the data the LM and Fast models are trained on).

    python examples/03_gym_generate.py
"""
from humm.gym.dataset import generate

for d in generate(3, "test", seed=0, langs=("en",)):
    print(f'[{d["query_type"]}, rung {d.get("rung")}]')
    print(d["context"])
    print("Q:", d["question"])
    answer = d["options"][d["label"]] if d["options"] else ("yes" if d["label"] == 1 else "no")
    print("A:", answer, f'(engine value {d["value"]})' if d.get("value") is not None else "", "\n")
