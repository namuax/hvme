import pytest
import pandas as pd

from humm.bench import prompts, runner


class FakeProvider:
    name = "fake"

    def option_probs(self, prompts_, labels):
        return [[0.8, 0.2] if "rains" in p else [0.3, 0.7] for p in prompts_]

    def complete(self, prompts_, max_new_tokens=1024):
        return ["...\nFinal answer: yes" if "rains" in p else "I think the answer is no." for p in prompts_]


DF = pd.DataFrame({"prompt": ["It rains. Is the grass wet?", "Is the sun cold?"], "label": [1, 0], "rung": [1, 2]})


def test_parse_yes_no():
    assert runner.parse_yes_no("Yes, because... Final answer: **no**") == 0
    assert runner.parse_yes_no("so the answer is yes.") == 1
    assert runner.parse_yes_no("unclear") == -1
    assert runner.parse_yes_no("there are no confounders, so P(Y|X) = 0.42") == -1  # a 'no' inside a truncated solution is not an answer


def test_noul_and_cot_summary():
    res = pd.concat([runner.run_noul(FakeProvider(), DF, prompts.cladder_noul), runner.run_cot(FakeProvider(), DF, prompts.cladder_cot)])
    table = runner.summarize(res)
    assert table.set_index("mode").loc["noul", "acc"] == 1 and table.set_index("mode").loc["cot", "acc"] == 1
    assert "ece" in table.columns


def test_caugym_and_calm_scoring():
    from humm.bench.datasets import BENCH
    from humm.bench.general import caugym_correct

    assert caugym_correct('풀이 {"ANSWER": "Yes", "PROB": "0.1234"}', '{"PROB": "0.12343"}') and not caugym_correct('{"PROB": "0.1250"}', '{"PROB": "0.1234"}')
    assert caugym_correct("LACK_CONDITION", "LACK_CONDITION")
    if not (BENCH / "calm_repo").exists():
        pytest.skip("CaLM repository not found (data/bench/calm_repo)")
    from humm.bench import calm as C

    assert len(C.TASKS) == 45
    q = C.questions("ATE-P_ATE-basic_EN", n=2)
    assert len(q) == 2 and C.correct("ATE-P_ATE-basic_EN", q[0], f'{{"ANSWER": "No", "PROB": "{q[0]["gt_answer"]:.4f}"}}')
    assert C.tail("풀이\n\n최종 답: yes") == "최종 답: yes" and C.tail("state yes\nFinal answer: No") == "No"
