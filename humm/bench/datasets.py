"""벤치마크 로더. 원본은 data/bench/ 에 한 번 내려받아 캐시한다."""
import urllib.request
from pathlib import Path

import networkx as nx
import pandas as pd

from ..paths import BENCH

DATA_DIR = BENCH
HF = "https://huggingface.co/datasets"
CAUGYM = "https://raw.githubusercontent.com/OpenCausaLab/CauGym/main/eval/test_datasets/{}_dataset.json"
CAUGYM_SETS = ("base", "rephrased", "omitted", "deconfounding", "redundant", "insufficient")

CORR2CAUSE_FILES = {
    "default": "test.csv",
    "dev": "dev.csv",
    "paraphrasing": "perturbation_by_paraphrasing_test.csv",
    "refactorization": "perturbation_by_refactorization_test.csv",
}


def _cached(url: str, name: str) -> Path:
    path = DATA_DIR / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def load_cladder(n_per_rung: int | None = None, seed: int = 0) -> pd.DataFrame:
    """CLadder v1.5 (Jin et al., NeurIPS 2023). label: 1=yes, 0=no. 질문이 포함된 full_v1.5_default를 쓴다."""
    df = pd.read_csv(_cached(f"{HF}/causal-nlp/CLadder/resolve/main/data/full_v1.5_default.csv", "cladder_full_v1.5.csv"))
    df = df.assign(label=(df.label == "yes").astype(int))
    return df if n_per_rung is None else df.groupby("rung").sample(n_per_rung, random_state=seed).reset_index(drop=True)


def load_corr2cause(variant: str = "default", n: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Corr2Cause (Jin et al., ICLR 2024). variant: default·dev·paraphrasing·refactorization. label: 1=함의됨.

    정답 1이 15~30%인 불균형 데이터이므로 정확도와 함께 F1을 본다.
    """
    name = CORR2CAUSE_FILES[variant]
    df = pd.read_csv(_cached(f"{HF}/causal-nlp/corr2cause/resolve/main/{name}", f"corr2cause_{name}"))[["input", "label"]]
    df = df.assign(label=(df.label == 1).astype(int))  # paraphrasing 변형은 라벨이 0/1/2이며, 1만 '함의됨'이다
    return df if n is None else df.sample(n, random_state=seed).reset_index(drop=True)


KMMLU_SUBJECTS = [
    "Accounting", "Agricultural-Sciences", "Aviation-Engineering-and-Maintenance", "Biology", "Chemical-Engineering", "Chemistry", "Civil-Engineering",
    "Computer-Science", "Construction", "Criminal-Law", "Ecology", "Economics", "Education", "Electrical-Engineering", "Electronics-Engineering",
    "Energy-Management", "Environmental-Science", "Fashion", "Food-Processing", "Gas-Technology-and-Engineering", "Geomatics", "Health",
    "Industrial-Engineer", "Information-Technology", "Interior-Architecture-and-Design", "Law", "Machine-Design-and-Manufacturing", "Management",
    "Maritime-Engineering", "Marketing", "Materials-Engineering", "Mechanical-Engineering", "Nondestructive-Testing", "Patent",
    "Political-Science-and-Sociology", "Psychology", "Public-Safety", "Railway-and-Automotive-Engineering", "Real-Estate", "Refrigerating-Machinery",
    "Social-Welfare", "Taxation", "Telecommunications-and-Wireless-Technology", "korean-history", "math",
]


def load_math500(n: int | None = None, seed: int = 0) -> pd.DataFrame:
    """MATH-500 (Lightman et al., 2023). 일반 능력(수학) 점검용. answer: 정답 문자열."""
    df = pd.read_json(_cached(f"{HF}/HuggingFaceH4/MATH-500/resolve/main/test.jsonl", "math500_test.jsonl"), lines=True)
    return df if n is None else df.sample(n, random_state=seed).reset_index(drop=True)


def load_kmmlu(n_per_subject: int = 7, seed: int = 0, split: str = "test") -> pd.DataFrame:
    """KMMLU (Son et al., 2024) 45과목에서 과목마다 n개. answer: 1~4(A~D). split 'train'은 SFT v2.3의 한국어 일반 궤적용."""
    dfs = {s: pd.read_csv(_cached(f"{HF}/HAERAE-HUB/KMMLU/resolve/main/data/{s}-{split}.csv", f"kmmlu/{s}-{split}.csv")) for s in KMMLU_SUBJECTS}
    return pd.concat([df.sample(min(n_per_subject, len(df)), random_state=seed).assign(subject=s) for s, df in dfs.items()], ignore_index=True)  # train은 1문항뿐인 과목이 있다


def load_bnlearn(name: str) -> nx.DiGraph:
    """bnlearn 정답 그래프(asia, sachs, alarm, insurance, child ...). pgmpy가 필요하다."""
    from pgmpy.example_models import load_model

    return nx.DiGraph(load_model(f"bnlearn/{name}").edges())


def load_caugym(name: str, n_per_type: int | None = None, seed: int = 0) -> pd.DataFrame:
    """CauGym 테스트(Chen et al., 2026). 유형 7개(ATE·CDE·ETT·NDE·NIE·PN·PS) × 100. answer는 {"ANSWER": …, "PROB": …} 또는 LACK_CONDITION(insufficient)."""
    df = pd.read_json(_cached(CAUGYM.format(name), f"caugym/{name}.json")).assign(set=name)
    df["qid"] = [f"{name}-{i}" for i in range(len(df))]
    return df if n_per_type is None else df.groupby("Type").sample(n_per_type, random_state=seed).sort_index()
