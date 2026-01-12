"""
🎯 PyTorch 실습: MNIST CNN 모델 학습 및 저장
1. CNN(Convolutional Neural Network) 모델 구조 이해
2. MNIST 손글씨 숫자 데이터셋으로 모델 학습
3. 학습된 모델 가중치를 .pth 파일로 저장
4. 저장된 모델을 FastAPI에서 로드하여 서빙하는 워크플로우의 첫 단계

📌 사전 준비:
pip install torch torchvision

📌 실행 방법(모델저장 위치를 위해서 디렉터리 이동후에: "cd lab3") 
python train_mnist.py

📌 생성되는 파일:
- mnist_cnn.pth (학습된 모델 가중치)
- ./data/MNIST/ (다운로드된 데이터셋)

💡 CNN 모델 구조:
입력 [1,28,28] → Conv2d → ReLU → MaxPool → Flatten → FC → FC → 출력 [10]
- Conv2d: 이미지에서 특징(엣지, 패턴) 추출
- MaxPool: 특징 맵 크기 축소, 중요 정보만 유지
- Flatten: 2D → 1D 변환 (FC 레이어 입력용)
- FC(Linear): 최종 분류 (0~9 숫자)

⚠️ 주의사항:
- 이 모델 클래스는 서빙 코드에서도 동일하게 정의해야 함!
- 1 Epoch만 학습 (실습용)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 1. CNN 모델 정의 (이미지 처리에 적합한 모델)
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        # 이미지 특징 추출 (Convolution)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3) 
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        # 분류 (Linear)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 13 * 13, 128)
        self.fc2 = nn.Linear(128, 10) # 0~9 숫자 분류

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train():
    # 데이터셋 준비 (없으면 다운로드함)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model = MNISTModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    print("🧠 학습 시작 (데이터가 많아 1~2분 걸릴 수 있습니다)...")
    
    # 빠르게 1 Epoch만 학습 (실습용)
    for batch_idx, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"진행률: {batch_idx}/{len(loader)}")

    # 모델 저장
    torch.save(model.state_dict(), "mnist_cnn.pth")
    print("✅ 모델 저장 완료: mnist_cnn.pth")

if __name__ == "__main__":
    train()