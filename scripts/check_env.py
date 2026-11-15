"""실습 환경 점검. 사용법: uv run scripts/check_env.py"""

import platform
import sys

import torch
import torchvision

from dlbasic import DATA_DIR, get_device


def main() -> None:
    device = get_device()
    cuda_build = torch.version.cuda  # CPU 빌드면 None
    build = f"CUDA {cuda_build}" if cuda_build else "CPU"

    print(f"OS           : {platform.system()} {platform.machine()}")
    print(f"Python       : {sys.version.split()[0]}")
    print(f"torch        : {torch.__version__}")
    print(f"torchvision  : {torchvision.__version__}")
    print(f"설치된 빌드   : {build}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
    print(f"MPS 사용 가능 : {torch.backends.mps.is_available()}")
    print(f"선택된 디바이스: {device}")
    print(f"데이터 경로   : {DATA_DIR}")

    if cuda_build and not torch.cuda.is_available():
        print(
            "\n[주의] CUDA 빌드를 설치했지만 GPU를 사용할 수 없습니다.\n"
            "  NVIDIA 드라이버 버전이 낮거나 NVIDIA GPU가 없는 경우입니다.\n"
            "  README의 '설치 옵션 고르기'와 '문제 해결'을 확인하세요."
        )

    # 선택된 디바이스에서 실제 연산이 되는지 확인한다.
    try:
        x = torch.randn(256, 256, device=device)
        y = (x @ x.T).sum().item()
    except RuntimeError as e:
        print(
            f"\n연산 테스트   : 실패\n  {e}\n"
            "  GPU와 설치 옵션이 맞지 않을 수 있습니다 (예: RTX 50 시리즈에 cu126 설치).\n"
            "  README의 '설치 옵션 고르기'를 확인하세요."
        )
        sys.exit(1)
    print(f"연산 테스트   : 통과 ({y:.2f})")


if __name__ == "__main__":
    main()
