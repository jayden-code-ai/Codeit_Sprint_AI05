# Codeit Sprint AI05 학습 기록

부트캠프에서 진행한 실습과 미션을 정리한 저장소입니다.  
포트폴리오에 활용하기 위해 학습 과정과 성장 흐름이 보이도록 구성했습니다.

## ✨ 핵심 요약
> - 부트캠프 4개 파트 실습/미션을 시간 흐름대로 정리한 기록
> - NLP/LLM 집중 학습: Transformer → BERT/GPT → PEFT/RAG
> - FastAPI/Streamlit로 모델 결과를 서비스 형태로 연결

## 파일들 보는 법
- `Part1_practice` ~ `Part4_practice`: 파트별 실습 내용을 모아 둔 폴더입니다.
- `Part1_mission` ~ `Part4_mission`: 파트별 미션/과제 결과물을 모아 둔 폴더입니다.
- `Python_basic`: 파이썬 기초 학습 자료입니다.

주요 하위 폴더 예시:
- `Part3_practice/01_자연어처리_실습`
- `Part3_practice/02_TransFormer_실습`
- `Part3_practice/03_BERT와GPT`
- `Part3_practice/04_ChatGPT_API실습`
- `Part3_practice/05_PEFT실습`
- `Part3_practice/06_RAG`
- `Part4_practice/02_FastAPI`

## 배운 기술스택
- 언어: Python
- NLP/LLM: 자연어처리, Transformer, BERT/GPT, ChatGPT API, PEFT, RAG
- 웹/API: 웹앱 프레임워크, FastAPI
- 실습 역량: 데이터 전처리, 모델 학습/평가, 간단한 서비스화 흐름

## 중요하게 보면 좋은 포인트
- **핵심 하이라이트**: `Part3_practice`에서 NLP/LLM 전처리 → 모델 → 응용 흐름이 이어집니다.
- **미션 집중 구간**: `Part3_mission/mission_10` ~ `mission_14`에 문제 정의/실험/정리 과정이 남아 있습니다.
- **서비스화 연결**: `Part4_practice/02_FastAPI`에서 모델 결과를 API로 확장하는 과정을 확인할 수 있습니다.
- **기초 다지기**: `Python_basic`에서 문법과 데이터 처리 기본기를 확인할 수 있습니다.

## ⭐ 대표 과제/실습 하이라이트
- **NLP 임베딩/분류**: [미션10 텍스트 임베딩/분류](Part3_mission/mission_10/미션10_1팀_정수범.ipynb) - 임베딩 기반 분류 흐름 정리
- **번역 모델**: [미션11 번역 Seq2Seq/Transformer](Part3_mission/mission_11/미션11_1팀_정수범_final.ipynb) - attention/transformer 비교 학습
- **문서 요약**: [미션12 문서 요약](<Part3_mission/mission_12/미션12_1팀 정수범_1st.ipynb>) - 요약 문제 설계와 결과 분석
- **PEFT 튜닝**: [미션13 PEFT 감성 분석](Part3_mission/mission_13/미션13_1팀_정수범_1st.ipynb) - LoRA 기반 파인튜닝 흐름
- **RAG 질의응답**: [미션14 RAG 기반 질의응답](Part3_mission/mission_14/미션14_1팀_정수범.ipynb) - 검색+응답 파이프라인 구성
- **RAG 챗봇 실습**: [RAG 실습: 여행가이드 챗봇](Part3_practice/06_RAG/251204_코드실습_여행가이드_챗봇.ipynb) - 간단한 챗봇 시나리오 구현
- **Streamlit 대시보드**: [Streamlit 대시보드 시작점](Part4_practice/01_웹앱프레임워크/streamlit-실습-1/src/1_dashboard.py) - UI 구성과 데이터 표시
- **FastAPI 기본 라우팅**: [FastAPI 기본 라우팅](Part4_practice/02_FastAPI/fastapi-실습-1/fastapi-실습-1/lab1_fastapi_basics.py) - 라우팅/요청 처리 기초
- **Streamlit+FastAPI 연동**: [Streamlit+FastAPI 연동](Part4_practice/02_FastAPI/fastapi-실습-3/fastapi-실습-3/frontend/app_streamlit.py) - 프론트/백엔드 연결 흐름

## ▶ 실행 방법
### Jupyter 노트북 (대부분의 실습/미션)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install jupyter
jupyter lab
```
- 필요한 패키지는 노트북 주제에 따라 다르므로 실행 중 부족한 패키지를 추가 설치합니다.

### Streamlit 실습
```bash
cd Part4_practice/01_웹앱프레임워크/streamlit-실습-1
pip install -r requirements.txt
streamlit run src/1_dashboard.py
```
- 다른 화면은 `src/2_tuning.py`, `src/3_model_serving.py` 등으로 실행합니다.

### FastAPI 실습
```bash
cd Part4_practice/02_FastAPI/fastapi-실습-1/fastapi-실습-1
pip install -r requirements.txt
uvicorn lab1_fastapi_basics:app --reload
```
- 실습 2/3은 각 폴더의 `README.md`를 참고합니다.
- 일부 실습은 OpenAI API 키 등 환경변수가 필요합니다.

## 지나온 성장 과정
- 파이썬 기초 → 데이터 처리와 문제 해결 감각을 확립
- NLP/Transformer → 언어 모델의 핵심 구조와 동작 원리 이해
- BERT/GPT/ChatGPT API → LLM 활용 실습으로 적용 능력 강화
- PEFT/RAG → 최신 모델 활용 방식과 응용 아이디어 확장
- FastAPI → 학습 결과를 서비스 형태로 연결하는 경험 축적
