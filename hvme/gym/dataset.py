"""Hvme-Gym dataset generation and storage.

- train/dev: training domains (marketing, manufacturing, medical); test: held-out domains (it_ops, meaningless names)
- Each split has its own random stream; items with identical context+question are deduplicated within and across splits.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import GYM
from .queries import GENERATORS
from .vocab import TEST_DOMAINS, TRAIN_DOMAINS

# Weights for nie, nde, ate and counterfactual queries are raised based on baseline analysis
MIX = {
    "ate": 0.14, "nie": 0.12, "nde": 0.10, "ett": 0.10, "cde": 0.06, "correlation": 0.08, "marginal": 0.06,
    "backadj": 0.10, "dsep": 0.06, "identify": 0.08, "corr2cause": 0.06, "ate_score": 0.04,
}
SPLIT_ID = {"train": 0, "dev": 1, "test": 2}


def _key(d: dict) -> str:
    return hashlib.sha1((d["context"] + "\n" + d["question"]).encode()).hexdigest()


def quotas(n: int, mix: dict) -> dict:
    w = np.array(list(mix.values()), float)
    q = np.floor(n * w / w.sum()).astype(int)
    q[np.argsort(-(n * w / w.sum() - q))[: n - q.sum()]] += 1
    return dict(zip(mix, q.tolist()))


def generate(n: int, split: str = "train", seed: int = 0, mix: dict = MIX, langs=("ko", "en"), exclude: set | None = None, generators: dict = GENERATORS) -> list[dict]:
    rng = np.random.default_rng([seed, SPLIT_ID[split]])
    domains = TEST_DOMAINS if split == "test" else TRAIN_DOMAINS
    seen = set(exclude or ())
    items = []
    for qtype, k in quotas(n, mix).items():
        made = 0
        for _ in range(k * 30):
            if made == k:
                break
            it = generators[qtype](rng, str(rng.choice(langs)), str(rng.choice(domains)))
            d = it and it.to_dict()
            if d and _key(d) not in seen:
                seen.add(_key(d))
                items.append(d | {"id": f"{split}-{qtype}-{made:06d}", "split": split})
                made += 1
    return [items[i] for i in rng.permutation(len(items))]


def _version(version: int) -> dict:
    """2: legacy generator, 3: hvme.gym.v3, 31: v3.1 (adds CLadder-style surface), 32: v3.2 (adds stateless-name templates), 33: v3.3 (corr2cause original-benchmark surface and tools), 34: v3.4 (numeric-query states 1/0), 35: v3.5 (swaps named treatment states)."""
    if version == 2:
        return {}
    from . import v3
    return {"mix": v3.MIX, "generators": {31: v3.GENERATORS_31, 32: v3.GENERATORS_32, 33: v3.GENERATORS_33, 34: v3.GENERATORS_34, 35: v3.GENERATORS_35}.get(version, v3.GENERATORS)}


def write_jsonl(items: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(d, ensure_ascii=False) + "\n" for d in items)
    return path


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def build(sizes: dict | None = None, seed: int = 0, out_dir: str | Path = GYM, version: int = 2) -> pd.DataFrame:
    """Builds train, dev, test in order; items in earlier splits are excluded from later ones. Returns per-split statistics. version=3 uses hvme.gym.v3, 31 uses the v3.1 generator."""
    sizes = sizes or {"train": 100_000, "dev": 2_000, "test": 5_000}
    seen, stats = set(), []
    for split, n in sizes.items():
        items = generate(n, split, seed, exclude=seen, **_version(version))
        seen |= {_key(d) for d in items}
        write_jsonl(items, Path(out_dir) / f"{split}.jsonl")
        stats.append(summary(items).assign(split=split))
    return pd.concat(stats)


def summary(items: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(items)
    return df.groupby(["query_type", "rung", "kind"]).agg(n=("id", "size"), label_mean=("label", "mean"), ko=("lang", lambda s: (s == "ko").mean())).reset_index()
