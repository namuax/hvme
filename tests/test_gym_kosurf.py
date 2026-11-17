"""Gym v3.10 Korean CLadder-style surface: same labels and values as the English item, and no English sentence left."""
import re

from hvme.gym import kosurf


def test_kosurf_pairs():
    en, ko = kosurf.generate_pairs(90, seed=3)
    assert len(en) == len(ko) > 80 and {d["query_type"] for d in ko} == set(kosurf.MIX_310)
    for e, k in zip(en, ko):
        assert (k["label"], k["value"], k["src_id"], k["lang"]) == (e["label"], e["value"], e["id"], "ko")
        text = k["context"] + " " + k["question"]
        assert not re.search(r"\b(not|the|and|is|of|For|We)\b", text) and "직접적인 영향을" in text and text.endswith("?")
