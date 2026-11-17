"""CaLM-lite(영어 45과제, 개방형 CEG-O 제외): 원 저장소(data/bench/calm_repo)의 프롬프트 생성·채점 함수를 그대로 쓴다.

채점은 응답 전체에 원 함수를 적용한 값(correct)과, 최종 답에만 적용한 값(correct_tail)을 함께 남긴다.
원 채점은 응답에서 처음 나오는 예/아니오·보기 문자열을 읽어서, 풀이를 먼저 쓰는 모델은 중간 문장이 답으로 읽힌다.
tool 모드에서는 변수 상태 yes/no가 먼저 나와 원 채점이 거의 늘 상태 이름을 읽는다. 최종 답은 마지막 'Final answer:' 뒤, 없으면 마지막 문단이다.
"""
import random
import re
import sys
from pathlib import Path

from .datasets import BENCH

REPO = BENCH / "calm_repo"
sys.path.insert(0, str(REPO / "calm"))

from data_processing.generate_questions import generate_question_list  # noqa: E402
from data_processing.task_hiearchy import task_hiearchy_dict  # noqa: E402
from evaluation.core_metrics import initialize_core_metric_evaluation_components, task_to_accuracy_module_map  # noqa: E402

TASKS = [t for t in task_hiearchy_dict if t.endswith("_EN") and task_to_accuracy_module_map[t] != "open-ended"]


def questions(task: str, style: str = "basic", n: int | None = None, seed: int = 0) -> list[dict]:
    """과제 하나의 문항(원 프롬프트 'question' 포함). n이면 시드 고정으로 n개만."""
    items = generate_question_list(dataset_path=REPO / "calm_lite_dataset" / f"{task_hiearchy_dict[task]}{task}.json", prompt_style=style)
    return items if n is None else sorted(random.Random(seed).sample(items, n), key=lambda d: d["index"])


def correct(task: str, item: dict, response: str, style: str = "basic") -> bool:
    """원 채점 함수로 한 문항의 정오를 낸다."""
    gt, pred, acc = initialize_core_metric_evaluation_components(task)
    return acc([gt(item)], [pred(response, item, style, task.split("-")[0])]) == 1


def tail(response: str) -> str:
    """최종 답: 마지막 'Final answer:' 뒤, 없으면 마지막 빈 줄 뒤의 문단."""
    i = response.lower().rfind("final answer:")
    return response[i + len("final answer:"):].strip() if i >= 0 else response.strip().split("\n\n")[-1]


GIVEN = re.compile(r"direct effect on|probability|causes|->", re.I)


def needs_tools(question: str) -> bool:
    """v2.12 경로 규칙: 인과 관계(그래프)나 확률이 주어진 문제에만 도구를 준다. 텍스트·상식 인과(COPA·e-CARE·ECI·CRASS 등)는 도구 없이 푼다.
    4B 평가표에서 v2.11 tool은 상식 인과에도 그래프를 억지로 세워 떨어졌다(CRASS 80 → 17)."""
    return bool(GIVEN.search(question))
