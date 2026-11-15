"""실습용 데이터셋을 미리 내려받는다.

사용법:
    uv run scripts/download_data.py            # 전체
    uv run scripts/download_data.py mnist      # 일부만
"""

import argparse

from torchvision import datasets

from dlbasic import DATA_DIR

DATASETS = {
    "mnist": datasets.MNIST,
    "fashion-mnist": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="실습용 데이터셋을 미리 내려받는다.")
    parser.add_argument("names", nargs="*", help=f"받을 데이터셋: {', '.join(DATASETS)} (생략 시 전체)")
    args = parser.parse_args()
    if unknown := set(args.names) - set(DATASETS):
        parser.error(f"알 수 없는 데이터셋: {', '.join(sorted(unknown))}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.names or DATASETS:
        print(f"[{name}] -> {DATA_DIR}")
        for train in (True, False):
            DATASETS[name](root=DATA_DIR, train=train, download=True)
    print("완료")


if __name__ == "__main__":
    main()
