"""Variable vocabulary. Each entry is (Korean name, Korean state 1, Korean state 0, English name, English state 1, English state 0).

Train/test split: training uses marketing, manufacturing and medical; testing uses it_ops and meaningless names (nonsense).
"""
import string

import numpy as np

DOMAINS = {
    "marketing": [
        ("광고 집행", "집행", "미집행", "ad campaign", "on", "off"),
        ("계절", "성수기", "비수기", "season", "peak", "off-peak"),
        ("할인 행사", "진행", "없음", "discount event", "on", "off"),
        ("웹 방문", "많음", "적음", "web visits", "high", "low"),
        ("장바구니 담기", "많음", "적음", "cart additions", "high", "low"),
        ("구매 전환", "높음", "낮음", "purchase conversion", "high", "low"),
        ("매출", "높음", "낮음", "sales", "high", "low"),
        ("재구매", "있음", "없음", "repurchase", "yes", "no"),
        ("고객 만족", "높음", "낮음", "customer satisfaction", "high", "low"),
        ("경쟁사 프로모션", "있음", "없음", "competitor promotion", "yes", "no"),
        ("SNS 언급", "많음", "적음", "social mentions", "high", "low"),
        ("브랜드 인지도", "높음", "낮음", "brand awareness", "high", "low"),
    ],
    "manufacturing": [
        ("설비 노후", "노후", "양호", "equipment age", "old", "new"),
        ("정비", "실시", "미실시", "maintenance", "done", "skipped"),
        ("온도", "높음", "정상", "temperature", "high", "normal"),
        ("습도", "높음", "정상", "humidity", "high", "normal"),
        ("진동", "큼", "작음", "vibration", "high", "low"),
        ("원자재 품질", "불량", "양호", "raw material quality", "poor", "good"),
        ("작업자 숙련도", "숙련", "미숙", "operator skill", "skilled", "unskilled"),
        ("라인 속도", "빠름", "보통", "line speed", "fast", "normal"),
        ("불량률", "높음", "낮음", "defect rate", "high", "low"),
        ("설비 고장", "발생", "없음", "equipment failure", "yes", "no"),
        ("생산량", "많음", "적음", "output", "high", "low"),
        ("납기 지연", "발생", "없음", "delivery delay", "yes", "no"),
    ],
    "medical": [
        ("흡연", "함", "안 함", "smoking", "yes", "no"),
        ("운동", "함", "안 함", "exercise", "yes", "no"),
        ("비만", "있음", "없음", "obesity", "yes", "no"),
        ("고혈압", "있음", "없음", "hypertension", "yes", "no"),
        ("당뇨", "있음", "없음", "diabetes", "yes", "no"),
        ("약물 복용", "복용", "미복용", "medication", "taken", "not taken"),
        ("콜레스테롤", "높음", "정상", "cholesterol", "high", "normal"),
        ("유전 소인", "있음", "없음", "genetic risk", "yes", "no"),
        ("연령", "고령", "청년", "age", "old", "young"),
        ("심장 질환", "있음", "없음", "heart disease", "yes", "no"),
        ("입원", "함", "안 함", "hospitalization", "yes", "no"),
        ("회복", "빠름", "느림", "recovery", "fast", "slow"),
    ],
    "it_ops": [
        ("트래픽", "급증", "정상", "traffic", "spike", "normal"),
        ("배포", "있음", "없음", "deployment", "yes", "no"),
        ("설정 변경", "있음", "없음", "config change", "yes", "no"),
        ("CPU 사용률", "높음", "정상", "CPU usage", "high", "normal"),
        ("메모리 누수", "있음", "없음", "memory leak", "yes", "no"),
        ("디스크 I/O", "높음", "정상", "disk I/O", "high", "normal"),
        ("응답 지연", "큼", "작음", "latency", "high", "low"),
        ("오류율", "높음", "낮음", "error rate", "high", "low"),
        ("장애", "발생", "없음", "outage", "yes", "no"),
        ("캐시 적중률", "낮음", "높음", "cache hit rate", "low", "high"),
        ("DB 잠금", "있음", "없음", "DB lock", "yes", "no"),
        ("알림", "발생", "없음", "alert", "yes", "no"),
    ],
}

TRAIN_DOMAINS = ["marketing", "manufacturing", "medical"]
TEST_DOMAINS = ["it_ops", "nonsense"]


def nonsense(n: int, rng: np.random.Generator) -> list[tuple]:
    """CLadder-style meaningless variable names, so the answer cannot be guessed from common sense."""
    names = set()
    while len(names) < n:
        names.add("".join(rng.choice(list(string.ascii_lowercase), 4)))
    return [(w, "1", "0", w, "1", "0") for w in sorted(names)]


def sample_vars(domain: str, n: int, rng: np.random.Generator) -> list[tuple]:
    """Draw n entries from the domain vocabulary. The order is random, so the graph may disagree with the commonsense causal direction (prevents commonsense shortcuts)."""
    pool = nonsense(n, rng) if domain == "nonsense" else DOMAINS[domain]
    return [pool[i] for i in rng.choice(len(pool), n, replace=False)]
