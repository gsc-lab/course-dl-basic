# course-dl-basic

영진전문대학교 글로벌시스템융합과 딥러닝 기초 실습용 예제와 데이터셋을 관리하는 프로젝트입니다. PyTorch를 사용하며, 환경은 [uv](https://docs.astral.sh/uv/)로 관리합니다.

- **대상:** 2학년 2학기
- **학점:** 4학점

## 설치

### 1. uv 설치

- macOS / Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

Python은 따로 설치하지 않아도 됩니다. uv가 이 프로젝트에 맞는 버전(3.12)을 자동으로 받습니다.

### 2. 설치 옵션 고르기

PyTorch는 계산에 쓸 장치에 따라 설치 파일이 다릅니다. 그래서 자기 컴퓨터에 맞는 옵션을 **하나** 골라야 합니다.

- **CPU:** 모든 컴퓨터에 있는 기본 연산 장치입니다. 어디서나 동작하지만 큰 모델 학습은 느립니다.
- **GPU:** 그래픽 카드입니다. 딥러닝 학습을 CPU보다 훨씬 빠르게 처리합니다.
- **CUDA:** NVIDIA 그래픽 카드로 계산하게 해 주는 기술입니다. **NVIDIA GPU에서만** 동작하며, Mac이나 AMD·Intel 그래픽에서는 쓸 수 없습니다.

아래 표에서 자기 컴퓨터에 해당하는 줄을 찾으세요.

| 내 컴퓨터 | 옵션 | CUDA |
|---|---|---|
| Mac (Apple Silicon, macOS 14 이상) | `cpu` | 사용 불가 |
| NVIDIA GPU가 없는 Windows / Linux | `cpu` | 사용 불가 |
| NVIDIA RTX 20 · 30 · 40 시리즈 | `cu126` | 사용 가능 |
| NVIDIA RTX 50 시리즈 | `cu130` | 사용 가능 |

- **Mac은 항상 `cpu`입니다.** Mac에는 NVIDIA GPU가 없어 CUDA를 쓸 수 없습니다. 대신 Apple Silicon Mac은 `cpu` 옵션으로 설치해도 Mac 전용 GPU 가속(MPS)이 자동으로 켜집니다. Intel Mac은 지원하지 않으므로 실습실 PC를 사용하세요.
- **처음에는 누구나 `cpu`로 시작해도 됩니다.** 초기 주차 실습은 CPU로 충분하고, CUDA 옵션은 내려받는 용량이 수 GB입니다. 나중에 옵션만 바꿔 다시 설치하면 되며 예제 코드는 고칠 필요가 없습니다.

#### 내 GPU 확인하기 (Windows / Linux)

터미널(Windows는 PowerShell)에서 아래 명령을 실행합니다.

```bash
nvidia-smi
```

- **표가 출력되면** NVIDIA GPU가 있는 것입니다. 표에서 두 가지를 확인합니다.
  - **GPU 이름** (예: `NVIDIA GeForce RTX 4060`): 숫자의 앞 두 자리가 시리즈입니다. `4060`은 40 시리즈, `5070`은 50 시리즈입니다.
  - **Driver Version** (예: `581.42`): 아래 최소 버전보다 낮으면 CUDA 옵션이 동작하지 않습니다.
- **명령을 찾을 수 없다고 나오면** NVIDIA GPU가 없거나 드라이버가 설치되지 않은 것입니다. `cpu`를 선택하세요.

| 옵션 | 필요한 드라이버 버전 |
|---|---|
| `cu126` | 561 이상 |
| `cu130` | 580 이상 |

드라이버 버전이 낮으면 [NVIDIA 드라이버 다운로드](https://www.nvidia.com/drivers)에서 최신 버전으로 올립니다. 실습실 PC처럼 직접 올릴 수 없는 경우에는 `cpu`를 선택하고 조교에게 알려 주세요.

### 3. 패키지 설치

프로젝트 폴더에서, 2단계에서 고른 옵션에 해당하는 명령 **하나**를 실행합니다.

```bash
uv sync --locked --extra cpu      # Mac, 또는 NVIDIA GPU가 없는 컴퓨터
uv sync --locked --extra cu126    # NVIDIA RTX 20 · 30 · 40 시리즈
uv sync --locked --extra cu130    # NVIDIA RTX 50 시리즈
```

옵션을 바꾸려면 다른 명령을 다시 실행하면 됩니다. 기존 PyTorch는 자동으로 교체됩니다.

> **주의: `--extra` 없이 `uv sync`만 실행하지 마세요.** 옵션을 생략하면 PyTorch가 설치되지 않고, 이미 설치되어 있었다면 제거됩니다. 그렇게 됐다면 위 명령을 다시 실행하면 복구됩니다.

### 4. 환경 점검

```bash
uv run scripts/check_env.py
```

마지막 줄에 `연산 테스트 : 통과`가 나오고, `선택된 디바이스`가 아래와 같으면 준비가 끝난 것입니다.

| 설치한 옵션 | 선택된 디바이스 |
|---|---|
| `cpu` (Apple Silicon Mac) | `mps` |
| `cpu` (그 외) | `cpu` |
| `cu126`, `cu130` | `cuda` |

### 문제 해결

| 증상 | 원인과 해결 |
|---|---|
| 설치 중 `doesn't have a source distribution or wheel for the current platform` | 이 컴퓨터에서 쓸 수 없는 옵션입니다. Mac에서 `cu126`이나 `cu130`을 고른 경우이며, `cpu`로 설치합니다. |
| `ModuleNotFoundError: No module named 'torch'` | `--extra` 없이 `uv sync`를 실행한 경우입니다. 3단계 명령을 다시 실행합니다. |
| CUDA 옵션으로 설치했는데 `선택된 디바이스`가 `cpu` | 드라이버 버전이 낮거나 NVIDIA GPU가 없습니다. `nvidia-smi`로 확인합니다. |
| `연산 테스트 : 실패` | GPU와 옵션이 맞지 않습니다. RTX 50 시리즈에 `cu126`을 설치한 경우이며, `cu130`으로 다시 설치합니다. |

## 실습 실행

```bash
uv run labs/week01_tensor/01_linear_regression.py   # 스크립트 실행
uv run jupyter lab                                  # 노트북 실행
```

## 데이터셋

```bash
uv run scripts/download_data.py          # 전체 (MNIST, Fashion-MNIST, CIFAR-10)
uv run scripts/download_data.py mnist    # 일부만
```

데이터는 `data/`에 저장되며 Git에는 올라가지 않습니다. 다른 위치(예: 실습실 공유 폴더)의 데이터를 쓰려면 `DLBASIC_DATA_DIR` 환경변수에 경로를 지정합니다.

## 폴더 구조

```
labs/          주차별 실습 예제
src/dlbasic/   공용 유틸 (get_device, set_seed, DATA_DIR)
scripts/       환경 점검, 데이터 다운로드
data/          데이터셋 (Git 제외)
outputs/       체크포인트, 로그 (Git 제외)
scratch/       자유롭게 실험하는 폴더 (Git 제외)
```

## 예제 작성 규칙

- 디바이스는 `dlbasic.get_device()`로 얻고 `.to(device)`만 사용합니다. `.cuda()`를 직접 쓰지 않습니다.
- 데이터 경로는 `dlbasic.DATA_DIR`을 사용합니다.
- 노트북은 출력을 지운 뒤 커밋합니다.

## 버전 정책

`torch==2.14.1`, `torchvision==0.29.1`로 고정되어 있습니다. 학기 중에는 변경하지 않으며, 변경할 때는 `pyproject.toml`의 `cpu`, `cu126`, `cu130` extra를 함께 수정합니다.
