import os
import subprocess
import sys
from pathlib import Path

# 생성기가 집합 순회(문자열 해시 순서)에 기대면 프로세스마다 다른 문항이 나온다(노트북 19에서 다시 쓴 문맥과 문항이 어긋났다)
SCRIPT = """
import hashlib
import numpy as np
from humm.gym import graphid, prob, v3
gens = v3.GENERATORS_34 | v3.GENERATORS_35 | prob.GENERATORS_37 | graphid.GENERATORS_39
h = hashlib.md5()
for k in sorted(gens):
    for s in range(8):
        it = gens[k](np.random.default_rng(s), "en", "it_ops")
        h.update(repr(it and (it.context, it.question, it.label)).encode())
print(h.hexdigest())
"""


def test_generators_same_across_processes():
    root = Path(__file__).resolve().parents[1]
    run = lambda seed: subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, cwd=root, check=True,
                                      env=os.environ | {"PYTHONHASHSEED": seed, "PYTHONPATH": str(root)}).stdout
    assert run("1") == run("2")
