"""Hume-Gym 데이터셋 생성·저장.

- train/dev: 학습 도메인(marketing·manufacturing·medical), test: 보류 도메인(it_ops·무의미 이름)
- 분할마다 난수 스트림을 분리하고, 문맥+질문이 같은 문항은 분할 안팎에서 중복을 제거한다.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import GYM
from .queries import GENERATORS
from .vocab import TEST_DOMAINS, TRAIN_DOMAINS

# 기준선 분석(reports/2026-10-24_baseline.md 6.4)에 따라 nie·nde·ate·반사실 비중을 높였다
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
    """2: 기존 생성기, 3: humm.gym.v3, 31: v3.1(CLadder식 표면 추가), 32: v3.2(상태 없는 이름 문형 추가), 33: v3.3(corr2cause 원 벤치 표면·도구), 34: v3.4(수치 질의 상태 1/0), 35: v3.5(named 처치 상태 맞바꿈)."""
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
    """train → dev → test 순서로 만들며, 앞 분할의 문항은 뒤 분할에서 제외한다. 분할별 통계표를 돌려준다. version=3은 humm.gym.v3, 31은 v3.1 생성기."""
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
