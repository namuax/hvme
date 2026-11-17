import os
import subprocess
import sys
from pathlib import Path

# if the generator relies on set iteration (string hash order), different processes produce different items (rewritten contexts then mismatched their items)
SCRIPT = """
import hashlib
import numpy as np
from hvme.gym import graphid, prob, v3
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
