import os
import subprocess
import sys

import pytest

from humm.gym.dataset import generate
from humm.gym.formats import reward
from humm.lm.data import looping, pick, template_examples
from humm.lm.solutions import solve


def test_solutions_correct_all_types():
    items = generate(480, "train", seed=11) + generate(240, "test", seed=11)
    assert {d["query_type"] for d in items} >= {"marginal", "ate", "ett", "nie", "dsep", "backadj", "identify", "corr2cause", "ate_score"}
    assert all(reward(d, solve(d)) == 1 for d in items)


def test_generation_reproducible_across_processes():
    """The same items come out even with a different string-hash seed (i.e. across processes)."""
    code = "import hashlib,json;from humm.gym.dataset import generate;print(hashlib.sha1(json.dumps(generate(120,'train',seed=1),ensure_ascii=False).encode()).hexdigest())"
    out = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=os.environ | {"PYTHONHASHSEED": s}).stdout for s in ("1", "2")}
    assert len(out) == 1


def test_pick_weights():
    items = generate(600, "train", seed=2)
    got = pick(items, 200, {"identify": 3.0})
    share = sum(d["query_type"] == "identify" for d in got) / len(got)
    assert len(got) == pytest.approx(200, abs=12) and share > 0.15
    ex = template_examples(got[:3])
    assert all(e["response"].endswith(("yes", "no")) or e["response"][-1].isalnum() for e in ex)


def test_looping():
    assert looping("\n".join(["Step: compute the value again"] * 5)) and not looping("a\nb\nc")
