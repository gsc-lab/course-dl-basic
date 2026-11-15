import os
from pathlib import Path

# src/dlbasic/paths.py 기준으로 두 단계 위가 프로젝트 루트다.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 실습실 공유 폴더처럼 다른 위치의 데이터셋을 쓰려면 DLBASIC_DATA_DIR 환경변수로 지정한다.
DATA_DIR = Path(os.environ.get("DLBASIC_DATA_DIR", PROJECT_ROOT / "data"))
OUTPUT_DIR = PROJECT_ROOT / "outputs"
