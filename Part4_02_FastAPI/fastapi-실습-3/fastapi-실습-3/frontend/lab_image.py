"""
🎯 FastAPI 실습: HuggingFace Vision Transformer 이미지 분류
1. Lifespan을 활용한 ViT(Vision Transformer) 모델 로딩
2. UploadFile과 File을 사용한 이미지 파일 업로드 처리
3. PIL을 활용한 바이트 데이터 → 이미지 객체 변환
4. 이미지 분류 결과 상위 3개 반환 패턴

📌 사전 준비:
1. pip install fastapi uvicorn transformers pillow python-multipart torch
2. 첫 실행 시 모델 다운로드 (약 350MB, google/vit-base-patch16-224)

📌 실행 방법:
python ./frontend/lab_image.py

💡 이미지 처리 흐름:
┌──────────┐    ┌───────────┐    ┌───────────┐    ┌──────────┐
│ 파일업로드 │ → │ 바이트읽기 │ → │ PIL 변환  │ → │ 모델추론 │
│ UploadFile│    │ file.read()│    │ Image.open│    │ classifier│
└──────────┘    └───────────┘    └───────────┘    └──────────┘

⚠️ 주의사항:
- python-multipart 필수! (파일 업로드 처리용)
- 이미지 크기가 크면 자동으로 224x224로 리사이즈됨
- GPU 없이도 동작하지만 CPU에서는 느릴 수 있음
"""

from fastapi import FastAPI, File, UploadFile
from contextlib import asynccontextmanager
from transformers import pipeline
from PIL import Image
import io

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("이미지 분류 모델 로딩 중...")
    # 이미지 분류(image-classification) 파이프라인 로드
    ml_models["vision_classifier"] = pipeline("image-classification", model="google/vit-base-patch16-224")
    print("모델 로딩 완료!")
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.post("/classify-image")
async def classify_image(file: UploadFile = File(...)):
    # 1. 업로드된 파일의 내용을 읽음
    image_data = await file.read()
    
    # 2. 바이트 데이터를 PIL 이미지 객체로 변환
    image = Image.open(io.BytesIO(image_data))
    
    # 3. 모델 추론
    classifier = ml_models["vision_classifier"]
    results = classifier(image)
    
    # 4. 상위 3개 결과만 깔끔하게 반환
    return {"filename": file.filename, "predictions": results[:3]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)