"""
🎯 FastAPI 실습: HuggingFace 감성 분석 모델 서빙
1. Lifespan을 활용한 Transformers 파이프라인 모델 로딩
2. sentiment-analysis 파이프라인으로 텍스트 감성 분류
3. Pydantic BaseModel을 사용한 요청 데이터 검증
4. POSITIVE/NEGATIVE 분류 및 확신도(score) 반환

📌 사전 준비:
1. pip install fastapi uvicorn transformers torch pillow
2. 첫 실행 시 모델 자동 다운로드 (distilbert-base-uncased-finetuned-sst-2-english)

📌 실행 방법:
python ./frontend/lab_sentiment.py

💡 감성 분석 결과 예시:
- "I love this!" → {"label": "POSITIVE", "score": 0.9998}
- "This is terrible" → {"label": "NEGATIVE", "score": 0.9995}

⚠️ 주의사항:
- 기본 모델은 영어 텍스트에 최적화되어 있음
- 한국어 감성 분석은 별도 한국어 모델 필요 (예: beomi/KcELECTRA)
"""

from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from transformers import pipeline

# 1. 모델을 전역 변수로 선언
ml_models = {}

# 2. Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("모델을 로딩 중입니다... (잠시만 기다려주세요)")
    # 감성 분석(sentiment-analysis) 파이프라인 로드
    ml_models["sentiment_analyzer"] = pipeline("sentiment-analysis")
    print("모델 로딩 완료!")
    yield
    # 앱 종료 시 실행될 코드 (여기선 리소스 해제할 게 없으므로 비워둠)
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

# 데이터 입력 형식을 정의
class TextRequest(BaseModel):
    text: str

@app.post("/analyze-sentiment")
def analyze_sentiment(request: TextRequest):
    # 로드된 모델 가져오기
    analyzer = ml_models["sentiment_analyzer"]
    
    # 모델 추론 실행
    result = analyzer(request.text)
    
    # 결과 반환 (label: POSITIVE/NEGATIVE, score: 확신도)
    return {"original_text": request.text, "result": result[0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)