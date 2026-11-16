"""1주차: 선형 회귀로 보는 PyTorch 학습의 전체 흐름.

사용법: uv run labs/week01_tensor/01_linear_regression.py

ML 수업에서 배운 단계가 PyTorch에서 각각 어떤 코드가 되는지 순서대로 보여 준다.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dlbasic import set_seed

set_seed(42)

# 1) 데이터셋 구성: 특징 5개, 샘플 1000개. 정답 w, b를 정해 두고 y = Xw + b + 노이즈로 만든다.
TRUE_W, TRUE_B = torch.tensor([0.05, 0.30, 0.02, -0.03, -0.04]), 1.0
X = torch.rand(1000, 5) * torch.tensor([150.0, 5.0, 25.0, 40.0, 30.0])  # 특징마다 스케일이 다르다
y = (X @ TRUE_W + TRUE_B + 0.2 * torch.randn(1000)).unsqueeze(1)  # shape (1000, 1)
X_train, X_test, y_train, y_test = X[:800], X[800:], y[:800], y[800:]  # 학습 800, 테스트 200

# 2) Rescaling(표준화): 평균과 표준편차는 학습 데이터에서만 구한다.
# 둘째 줄을 주석 처리하고 실행하면 손실이 nan으로 발산한다.
mean, std = X_train.mean(dim=0), X_train.std(dim=0)
X_train, X_test = (X_train - mean) / std, (X_test - mean) / std

# DataLoader는 데이터를 섞어서 미니배치(32개씩)로 꺼내 준다.
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

# 3) 모델(가설): y_hat = Xw + b
model = nn.Linear(5, 1)  # 입력 5개, 출력 1개

# 4) 손실 함수: 평균 제곱 오차(MSE)
loss_fn = nn.MSELoss()

# 5) 경사하강법
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # lr: 학습률

for epoch in range(1, 21):
    for xb, yb in train_loader:
        loss = loss_fn(model(xb), yb)  # 예측하고 손실을 계산한다
        optimizer.zero_grad()  # 이전 배치의 기울기를 지운다
        loss.backward()  # 손실을 w, b로 미분한다 (기울기 계산)
        optimizer.step()  # w <- w - lr * 기울기
    if epoch % 5 == 0:
        print(f"epoch {epoch:2d} | loss {loss.item():.4f}")

# 6) 평가: 기울기가 필요 없으므로 torch.no_grad() 안에서 계산한다.
with torch.no_grad():
    print(f"test loss {loss_fn(model(X_test), y_test).item():.4f}")

    # 표준화된 입력으로 학습했으므로 w, b를 원래 단위로 되돌려 정답과 비교한다.
    w = model.weight.squeeze() / std
    b = model.bias - (w * mean).sum()
print("정답: w =", [round(v, 2) for v in TRUE_W.tolist()], "| b =", TRUE_B)
print("학습: w =", [round(v, 2) for v in w.tolist()], "| b =", round(b.item(), 2))
