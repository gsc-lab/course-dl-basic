"""1주차: 텐서와 자동 미분 기초.

사용법: uv run labs/week01_tensor/02_tensor_basics.py

01_linear_regression.py에서 쓴 텐서, backward(), optimizer.step()이 실제로 무엇인지 들여다본다.
"""

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

# 자동 미분: y = x^2 의 x=3 에서의 기울기
# requires_grad=True인 텐서는 자신에게 적용된 연산을 기록하고, backward()가 그 기록을 따라 미분한다.
x = torch.tensor(3.0, requires_grad=True)
y = x**2
y.backward()
print("dy/dx at x=3:", x.grad.item())

# 경사하강법 직접 구현: y = 2x + 1 을 만족하는 w, b 찾기
# 01에서 optimizer가 대신 해 준 일을 수업에서 배운 갱신 식 그대로 써 본다.
xs = torch.linspace(-1, 1, 20)
ys = 2 * xs + 1
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
lr = 0.1

for step in range(1, 201):
    loss = ((w * xs + b - ys) ** 2).mean()  # MSE
    loss.backward()  # w.grad, b.grad에 기울기가 채워진다
    with torch.no_grad():  # 갱신 자체는 미분 기록에 남기지 않는다
        w -= lr * w.grad  # optimizer.step()이 하는 일
        b -= lr * b.grad
    w.grad.zero_()  # optimizer.zero_grad()가 하는 일: 지우지 않으면 기울기가 계속 누적된다
    b.grad.zero_()
    if step % 50 == 0:
        print(f"step {step:3d} | loss {loss.item():.6f} | w {w.item():.3f} | b {b.item():.3f}")
