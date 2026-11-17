"""Paths relative to the working folder. Defaults to the repository root; when installed as a package, set the HUMM_HOME environment variable to locate the data, model and results folders."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("HUMM_HOME", Path(__file__).resolve().parents[1]))
DATA = ROOT / "data"
BENCH = DATA / "bench"
GRAPHS = DATA / "graphs"
RESULTS = ROOT / "results"
GYM = DATA / "gym"
GYM_V2 = DATA / "gym_v2"  # holds the vocabulary (meta.vocab) and is reproducible across processes. v1 (GYM) is left unchanged as the reference for earlier results
GYM_V3 = DATA / "gym_v3"  # SFT v2: more varied sentence patterns, CLadder-style context, det_cf (humm.gym.v3)
GYM_V31 = DATA / "gym_v31"  # SFT v2.1: v3 + CLadder-style surface forms (nonsense variables, not X, conjunction patterns)
GYM_V32 = DATA / "gym_v32"  # GRPO: v3.1 + stateless-name patterns (affect heart condition)
GYM_V33 = DATA / "gym_v33"  # SFT v2.2: v3.2 + original corr2cause benchmark surface forms (3-6 variables) and the causal_discovery tool
