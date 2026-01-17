"""
🎯 Streamlit 실습: FastAPI 백엔드와 연동하는 AI 데모 프론트엔드
1. Streamlit을 활용한 간단한 웹 UI 구축
2. requests 라이브러리로 FastAPI 서버와 통신
3. 텍스트 감성 분석(NLP)과 이미지 분류(Vision) 탭 구성
4. 파일 업로드 및 결과 시각화 패턴 학습

📌 사전 준비:
1. pip install streamlit requests pillow
2. 백엔드 서버 2개 실행 필요:
   - 감성 분석: uvicorn lab_sentiment:app --reload --port 8000
   - 이미지 분류: uvicorn lab_image:app --reload --port 8001

📌 실행 방법:
streamlit run ./frontend/app_streamlit.py

📌 접속 주소:
http://localhost:8501 (Streamlit 기본 포트)

💡 전체 아키텍처:
┌─────────────────┐      ┌─────────────────┐
│   Streamlit     │      │    FastAPI      │
│   (Frontend)    │ ───▶ │   (Backend)     │
│   Port: 8501    │ ◀─── │  Port: 8000/8001│
└─────────────────┘      └─────────────────┘
     브라우저 UI             ML 모델 서빙

⚠️ 주의사항:
- Streamlit은 스크립트가 변경되면 자동 재실행됨
- 백엔드 서버가 꺼져 있으면 "연결 실패" 에러 발생
"""

import streamlit as st
import requests
from PIL import Image
import io

# 페이지 설정
st.set_page_config(page_title="AI Model Demo", layout="wide")
st.title("FastAPI x Hugging Face 실습")

# 탭으로 기능 분리
tab1, tab2 = st.tabs(["📝 감성 분석 (NLP)", "🖼️ 이미지 분류 (Vision)"])

# --- 탭 1: 텍스트 감성 분석 ---
with tab1:
    st.header("이 문장은 긍정일까 부정일까?")

    # 1. 사용자 입력 받기
    user_input = st.text_area("영어로 문장을 입력해보세요:", "I am so happy to learn FastAPI!")

    if st.button("분석하기", key="text_btn"):
        if user_input:
            try:
                # 2. FastAPI 서버로 요청 보내기 (Backend Port: 8000)
                response = requests.post(
                    "http://localhost:8000/analyze-sentiment",
                    json={"text": user_input}
                )

                # 3. 응답 처리
                if response.status_code == 200:
                    result = response.json()["result"]
                    label = result["label"]
                    score = result["score"]

                    # 4. 결과 출력
                    if label == "POSITIVE":
                        st.success(f"😊 긍정적 문장입니다! (확신도: {score:.2f})")
                    else:
                        st.error(f"😞 부정적 문장입니다! (확신도: {score:.2f})")

                    # JSON 원본 출력 (학습용)
                    with st.expander("개발자용 원본 데이터 확인"):
                        st.json(response.json())

                else:
                    st.error("서버 에러가 발생했습니다.")
            except Exception as e:
                st.error(f"연결 실패! FastAPI 서버(8000번 포트)가 실행 중인지 확인하세요.\n오류 내용: {e}")

# --- 탭 2: 이미지 분류 ---
with tab2:
    st.header("이 이미지는 무엇일까?")

    # 1. 파일 업로더
    uploaded_file = st.file_uploader("이미지 파일을 업로드하세요 (jpg, png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # 업로드한 이밎 화면에 표시
        image = Image.open(uploaded_file)
        st.image(image, caption="업로드된 이미지", width=300)
        
        if st.button("이미지 분석하기", key="image_btn"):   
            try:
                # 2. 파일을 바이너리 형태로 변환하여 전송 준비
                # 중요: 스트림 위치를 처음으로 되돌림 (파일을 이미 읽었을 수도 있으므로)
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}

                # 3. FastAPI 서버로 요청 보내기 (Backend Port: 8001)
                response = requests.post(
                    "http://localhost:8001/classify-image",
                    files=files
                )

                # 4. 결과 처리
                if response.status_code == 200:
                    predictions = response.json()["predictions"]

                    st.subheader("분석 결과 Top 3")
                    for pred in predictions:
                        # 프로그레스 바로 시각화
                        st.write(f"**{pred['label']}** ({pred['score']*100:.1f}%)")
                        st.progress(pred['score'])
                else:
                    st.error("서버 에러가 발생했습니다.")
                
            except Exception as e:
                st.error(f"연결 실패! FastAPI 서버(8001번 포트)가 실행 중인지 확인하세요.\n오류 내용: {e}")