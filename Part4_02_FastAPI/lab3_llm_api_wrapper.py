import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI

# 환경변수 로드
load_dotenv()

app = FastAPI(title="나만의 LLM API 서버")

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1단계 : 기본 채팅 완성 API 
class Message(BaseModel):
    role: str = Field(pattern=r"^(user|assistant|system)$")
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = "gpt-4o-mini"
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=1000, ge=1, le=4096)

    model_config = {
        "json_schema_extra":{
            "examples": [{
                "messages": [
                    {"role": "user", "content": "안녕하세요!"}
                ]
            }]
        }
    }

class ChatResponse(BaseModel):
    response: str
    model: str
    usage: dict

@app.get("/")
def home():
    """서버 상태 확인"""
    return {"message": "LLM API 서버가 실행중입니다..."}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    기본 채팅 API

    💡 OpenAI API를 그대로 노출하지 않고 래핑하는 장점:
    1. API 키 보호 (클라이언트에 키 노출 안 함)
    2. 요청/응답 형식 커스터마이징
    3. 로깅, 모니터링 추가 가능
    4. 비용 제어 (max_tokens 제한 등)
    """
    try:
        response = client.chat.completions.create(
            model=request.model,
            messages=[m.model_dump() for m in request.messages],
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        return ChatResponse(
            response=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))