import json

import numpy as np
import pytest

from humm.gym import GENERATORS, generate, option_labels, parse_final, render, reward, target_index, target_text
from humm.gym.check import audit
from humm.gym.graphs import all_dags, mec_index
from humm.gym.verbalize import josa
from humm.gym.vocab import DOMAINS, TRAIN_DOMAINS


@pytest.fixture(scope="module")
def items():
    return generate(600, "train", seed=7)


def test_josa():
    assert [josa(w, "이") for w in ["매출", "광고", "L", "A", "회복"]] == ["매출이", "광고가", "L이", "A가", "회복이"]
    assert josa("집행", "로") == "집행으로" and josa("높음", "로") == "높음으로" and josa("빠름", "로") == "빠름으로" and josa("1", "로") == "1로"
    assert josa("미실시", "로") == "미실시로"


def test_markov_equivalence_counts():
    # known values: 3 nodes have 25 DAGs and 11 equivalence classes, 4 nodes have 543 DAGs and 185 classes
    assert (len(all_dags(3)), len(mec_index(3)), len(all_dags(4)), len(mec_index(4))) == (25, 11, 543, 185)


def test_every_generator_produces(items):
    assert set(d["query_type"] for d in items) == set(GENERATORS)


def test_labels_match_engine(items):
    assert audit(items) == []


def test_noul_labels_balanced(items):
    noul = [d["label"] for d in items if d["kind"] == "noul"]
    assert 0.4 < np.mean(noul) < 0.6


def test_numeric_effects_have_margin(items):
    v = [abs(d["value"] - (0.5 if d["query_type"] == "marginal" else 0)) for d in items if d["query_type"] in ("ate", "nie", "nde", "ett", "marginal")]
    assert min(v) >= 0.05


def test_test_split_uses_held_out_vocab():
    train_words = {w[0] for dom in TRAIN_DOMAINS for w in DOMAINS[dom]}
    test = generate(150, "test", seed=7)
    assert not any(w in d["context"] for d in test for w in train_words if len(w) > 2)


def test_reproducible_and_json_safe():
    a, b = generate(50, "dev", seed=3), generate(50, "dev", seed=3)
    assert a == b
    assert json.loads(json.dumps(a[0], ensure_ascii=False)) == a[0]


def test_formats_and_reward(items):
    for kind in ("noul", "choice", "score"):
        d = next(x for x in items if x["kind"] == kind)
        assert option_labels(d)[target_index(d)].lower() in (target_text(d).lower(), "예", "아니오")
        assert render(d, "cot").rstrip().endswith("'.") or "Final answer" in render(d, "cot")
        assert reward(d, f"풀이...\nFinal answer: {target_text(d)}") == 1.0
        assert reward(d, "풀이가 중간에 잘림") == 0.0
    assert parse_final("Final answer: **예**") == "yes" and parse_final("final answer: c") == "C"
