import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """random, numpy, torch의 난수 시드를 한 번에 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
