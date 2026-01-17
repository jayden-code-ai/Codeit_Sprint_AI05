# FastAPI 실습 3: Streamlit + HuggingFace 연동

FastAPI 백엔드와 Streamlit 프론트엔드를 연결하여 AI 모델을 서빙하는 실습입니다.

## 프로젝트 구조

```
frontend/
├── app_streamlit.py    # Streamlit UI (포트 8501)
├── lab_sentiment.py    # 감성 분석 API (포트 8000)
├── lab_image.py        # 이미지 분류 API (포트 8001)
└── cat.png             # 테스트 이미지
```

## 설치

```bash
pip install fastapi uvicorn streamlit requests transformers torch pillow python-multipart
```

## 실행 방법

**1. 백엔드 서버 실행 (터미널 2개 필요)**

```bash
# 터미널 1: 감성 분석 서버
uvicorn lab_sentiment:app --reload --port 8000

# 터미널 2: 이미지 분류 서버
uvicorn lab_image:app --reload --port 8001
```

**2. 프론트엔드 실행**

```bash
streamlit run ./frontend/app_streamlit.py
```

**3. 브라우저에서 확인**

http://localhost:8501 접속

## 주요 기능

| 탭 | 기능 | 모델 |
|---|---|---|
| 📝 감성 분석 | 영어 문장의 긍정/부정 분류 | distilbert-sst-2 |
| 🖼️ 이미지 분류 | 이미지 내 객체 인식 | google/vit-base-patch16-224 |

## 학습 포인트

- Lifespan을 활용한 ML 모델 로딩 패턴
- Streamlit ↔ FastAPI 통신 (requests 라이브러리)
- 파일 업로드 처리 (UploadFile, python-multipart)
- HuggingFace pipeline API 활용