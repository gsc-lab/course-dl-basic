import torch


def get_device() -> torch.device:
    """사용 가능한 가장 빠른 디바이스를 반환한다 (cuda > mps > cpu).

    예제 코드는 `.cuda()`를 직접 쓰지 않고 `.to(get_device())`만 사용한다.
    그래야 CPU 실습 코드가 CUDA 환경에서 수정 없이 동작한다.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
