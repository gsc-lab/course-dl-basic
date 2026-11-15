"""1주차: 텐서 기초. 사용법: uv run labs/week01_tensor/01_tensor_basics.py"""

import torch

from dlbasic import get_device, set_seed

set_seed(42)
device = get_device()
print(f"device: {device}")

# 텐서 생성
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.randn(2, 2)
print("a =", a, sep="\n")
print("shape:", a.shape, "| dtype:", a.dtype)

# 연산: 원소별 덧셈, 행렬 곱
print("a + b =", a + b, sep="\n")
print("a @ b =", a @ b, sep="\n")

# 디바이스 이동: .cuda() 대신 항상 .to(device)를 쓴다
a = a.to(device)
print("a.device:", a.device)

# 자동 미분 맛보기: y = x^2 의 x=3 에서의 기울기
x = torch.tensor(3.0, requires_grad=True)
y = x**2
y.backward()
print("dy/dx at x=3:", x.grad.item())
