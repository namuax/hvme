from humm.fast.distill import answer_index, distill_items, numbers_kept, paraphrased, storied, story2_prompt, story_prompt, target_from, verified
from humm.gym import graphid
from humm.gym.dataset import generate

OK = "Let me compute.\n<tool_call>\n<function=causal_engine>\n</function>\n</tool_call>\n<tool_response>\nATE = 0.12\n</tool_response>\nSo yes.\nFinal answer: yes"
ERR = OK.replace("ATE = 0.12", "error: unknown variable 'kwox'")


def _items():
    return generate(40, "train", seed=1, generators=graphid.GENERATORS_39, mix={"ate": 1, "identify": 1}, langs=("en",))


def test_numbers_kept_and_paraphrased():
    d = next(x for x in _items() if x["query_type"] == "ate")
    assert numbers_kept(d["context"], "x " + d["context"])
    assert not numbers_kept(d["context"], d["context"] + " The probability of zorp is 9137%.")  # 지어낸 숫자
    dropped = d["context"].replace(next(t for t in d["context"].split() if t[0].isdigit()), "many")
    assert paraphrased(d, dropped) is None
    p = paraphrased(d, "  " + d["context"] + " ")
    assert p["id"] == d["id"] + "-p" and p["label"] == d["label"] and p["meta"]["orig_context"] == d["context"]


def test_storied_keeps_numbers_and_orig_label():
    d = next(x for x in _items() if x["kind"] == "noul" and x["query_type"] == "ate")
    assert d["question"] in story_prompt(d)
    ok = f"Some preface.\nPassage: In a farm study, {d['context']}\nQuestion: Does rain help the harvest?\n\nNote: extra."
    s = storied(d, ok)
    assert s["id"] == d["id"] + "-s" and s["question"] == "Does rain help the harvest?" and s["meta"]["orig_label"] == d["label"]
    assert storied(d, ok.replace("Passage:", "Text:")) is None  # 형식이 다르다
    assert storied(d, ok + "\nQuestion: Does rain help the harvest?")["question"] == "Does rain help the harvest?"
    assert storied(d, ok + "\nQuestion: Does rain hurt the harvest?") is None  # 질문을 다르게 두 번
    assert storied(d, ok.replace("Question: Does", "Question: Is 9137% enough, does")) is None  # 지어낸 숫자


def test_verified_drops_engine_errors_and_loops():
    assert verified(OK) == "yes"
    assert verified(ERR) is None
    assert verified("Final answer: no") == "no"  # 도구를 부르지 않는 문항(det-cf 손풀이)
    assert verified("\n".join(["the same long line again"] * 5) + "\nFinal answer: no") is None


def test_targets_follow_verified_answers():
    its = _items()
    noul, choice = next(x for x in its if x["kind"] == "noul"), next(x for x in its if x["kind"] == "choice")
    assert answer_index(noul, "no") == 1 and answer_index(choice, "B") == 1 and answer_index(noul, "B") is None
    t = target_from(noul, ["yes", "yes", "no", None], alpha=0.0)
    assert t == [2 / 3, 1 / 3]
    assert target_from(noul, [None, None]) is None
    samples = [{"id": noul["id"], "output": OK}] * 3 + [{"id": noul["id"], "output": ERR}, {"id": choice["id"], "output": ERR}]
    out = distill_items([noul, choice], samples, alpha=0.1)
    assert [d["id"] for d in out] == [noul["id"]]  # 검증된 풀이가 없는 문항은 버린다
    assert out[0]["n_samples"] == 4 and out[0]["n_verified"] == 3 and abs(out[0]["target"][0] - 3.1 / 3.2) < 1e-9


def test_story2_rotates_and_counts():
    d = next(x for x in _items() if x["kind"] == "noul")
    assert "in medicine" in story2_prompt(d, 0) and "in farming" in story2_prompt(d, 1) and "a lab notebook entry" in story2_prompt(d, 12)
    assert not numbers_kept("five vars 5. P is 40%.", "five vars. P is 40%.") and numbers_kept("five vars 5. P is 40%.", "five vars. P is 40%.", counts=False)
    assert not numbers_kept("P is 40%.", "P is 4%.", counts=False) and not numbers_kept("P is 0.4.", "P is 0.5.", counts=False)
