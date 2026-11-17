"""작업 폴더 기준 경로. 기본은 저장소 루트이고, 패키지로 설치했으면 환경 변수 HUMM_HOME으로 데이터·모델·결과 폴더의 위치를 정한다."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("HUMM_HOME", Path(__file__).resolve().parents[1]))
DATA = ROOT / "data"
BENCH = DATA / "bench"
GRAPHS = DATA / "graphs"
RESULTS = ROOT / "results"
GYM = DATA / "gym"
GYM_V2 = DATA / "gym_v2"  # Step 9: 어휘(meta.vocab)를 담고 프로세스 간 재현되는 판. v1(GYM)은 Step 3·8 결과의 기준이라 그대로 둔다
GYM_V3 = DATA / "gym_v3"  # Step 9 SFT v2: 문형 다양화·CLadder식 문맥·det_cf (humm.gym.v3)
GYM_V31 = DATA / "gym_v31"  # Step 9 SFT v2.1: v3 + CLadder식 표면(무의미 변수, not X, 결합 문형)
GYM_V32 = DATA / "gym_v32"  # Step 9 GRPO: v3.1 + 상태 없는 이름 문형(affect heart condition)
GYM_V33 = DATA / "gym_v33"  # Step 9 SFT v2.2: v3.2 + corr2cause 원 벤치 표면(변수 3–6개)·causal_discovery 도구
