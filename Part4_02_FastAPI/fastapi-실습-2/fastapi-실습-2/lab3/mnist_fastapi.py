"""
🎯 FastAPI 실습: PyTorch MNIST CNN 모델 서빙
1. 직접 학습한 PyTorch 모델(.pth)을 로드하여 API로 서빙
2. 모델 클래스 정의가 학습 코드와 동일해야 하는 이유 이해
3. CNN 입력 형태 변환 (1D 리스트 → 4D 텐서) 전처리 패턴
4. Softmax를 활용한 확률 기반 예측 및 confidence 반환

📌 사전 준비:
1. pip install torch
2. 학습된 모델 파일 준비: mnist_cnn.pth (같은 디렉토리에 위치)

📌 실행 방법:
python mnist_fastapi.py


💡 입력 데이터 형태 변환 과정:
- 입력: [784] - 1차원 리스트 (28×28 펼친 것)
- 변환: [1, 1, 28, 28] - [배치, 채널, 높이, 너비]
- CNN은 반드시 4차원 텐서를 입력으로 받음!

⚠️ 주의사항:
- 모델 클래스(MNISTModel)는 학습 시 사용한 코드와 100% 동일해야 함
- 픽셀값은 0.0~1.0 사이로 정규화된 값이어야 함
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import torch
import torch.nn as nn

# --- 1. 모델 클래스 정의 (유의: 학습한 코드와 완전히 동일해야 함) ---
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32*13*13, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# --- 2. 전역 변수 ---
ml_models = {}

# --- 3. Lifespan (모델 로드) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("===== 서버 시작: MNIST 모델 로딩 중 ...")
    try:
        model = MNISTModel()

        from pathlib import Path
        MODEL_PATH = Path(__file__).parent / "mnist_cnn.pth"
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        ml_models["mnist"] = model
        print("✅ MNIST 모델 로드 성공!")

    except Exception as e:
        print(f"!!! 모델 로드 실피: {e}")
        ml_models["mnist"] = None

    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

# --- 4. 입력 스키마 ---
class ImageRequest(BaseModel):
    # 28x28 = 784개의 픽셀 값 (0.0 ~ 1.0 사이의 흑백 강도)
    pixels: list[float] = Field(..., min_length=784, max_length=784)

# --- 5. 추론 API ---
@app.post("/predict/digit")
async def predict_digit(request: ImageRequest):
    model = ml_models.get("mnist")
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # [전처리]
    # 1. 리스트 -> 텐서 변환
    input_tensor = torch.tensor(request.pixels, dtype=torch.float32)

    # 2. 형태 변환 (Reshape): [784] -> [배치크기 1, 채널 1, 높이 28, 너비 28]
    # CNN 모델은 4차원 입력 해야함
    input_tensor = input_tensor.view(1,1,28,28)

    # [추론]
    with torch.no_grad():
        logits = model(input_tensor)
        # Softmax를 거쳐 확률로 변환
        prob = torch.nn.functional.softmax(logits, dim=1)

    # [후처리]
    # 가장 높은 확률을 가진 숫자의 인덱스(.argmax())와 그 확률값(.max()) 가져오기
    predicted_class = prob.argmax().item()
    confidence = prob.max().item()

    return {
        "prediction": predicted_class,              # 예측된 숫자(0-9)
        "confidence": f"{confidence*100:.2f}%"      # 확신도
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)