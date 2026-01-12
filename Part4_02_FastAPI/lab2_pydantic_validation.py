from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime

app = FastAPI(title="Pydantic 데이터 검증 실습")

# 1단계 : 기본 Pydantic 모델

class UserCreate(BaseModel):
    """사용자 생성 요청 모델"""
    username : str
    email : str
    age : int

    # 예시 데이터 (Swagger UI에서 자동으로 표시됨)
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "username": "홍길동",
                    "email": "hong@example.com",
                    "age": 25
                }
            ]
        }
    }

class UserResponse(BaseModel):
    """사용자 응답 모델"""
    id : int
    username : str
    email : str
    age : int
    created_at : str

# 간단한 인메모리 저장소
fake_db = []
user_is_counter = 1

@app.post("/users", response_model=UserResponse)
def create_user(user: UserCreate):
    """
    사용자 생성 API

    # pydantic이 자동으로 해주는 것:
    1. json >> python 객체 변환
    2. 타입 검증(age에 문자열 넣으면 에러)
    3. 필수 필드 확인
    """
    global user_is_counter

    new_user = {
        "id": user_is_counter,
        "username": user.username,
        "email": user.email,
        "age": user.age,
        "created_at": datetime.now().isoformat()
    }
    fake_db.append(new_user)
    user_is_counter += 1

    return new_user

@app.get("/users/{user_id}", response_model=List[UserResponse])
def get_user():
    """저장된 사용자 목록 조회"""
    return fake_db

# 2단계 : 필드 검증 (Field Validation)
class ProductCreate(BaseModel):
    """상품 등록 모델 - 필드 검증 포함"""
    name: str = Field(
        min_length=2,   # 최소 2자
        max_length=50,  # 최대 50자
        description="상품명"
    )

    price: int = Field(
        gt=0,           # 0보다 큰 값
        le=10000000,  # 10,000,000 이하
        description="상품 가격"
    )

    quantity: int = Field(
        ge=0,          # 0 이상
        default=0,
        description="재고 수량"
    )

    category: str = Field(
        pattern=r"^(전자제품|의류|식품|기타)$",  # 정규식 패턴
        description="상품 카테고리"
    )

    description: Optional[str] = Field(
        default=None,
        max_length=500,
        description="상품 설명(선택)"
    )

# 상품 저장소
products_db = []
product_id_counter = 1

@app.post("/products")
def create_product(product: ProductCreate):
    """
    상품 생성 API

    테스트 해보기:
    1. price에 -100 입력 → 422 에러
    2. name에 1글자 입력 → 422 에러
    3. category에 '가구' 입력 → 422 에러
    """
    global product_id_counter

    new_product = {
        "id": product_id_counter,
        **product.model_dump(), # Pydantic 모델을 dict로 변환 후, 응답 dict에 그대로 병합
        "created_at": datetime.now().isoformat()
    }
    products_db.append(new_product)
    product_id_counter += 1

    return {
        "message": "상품 등록 성공",
        "product": new_product
    }

@app.get("/products")
def get_products():
    """상품 목록 조회"""
    return {"products": products_db, "total": len(products_db)}

# 3단계 : 커스텀 Validator
class ChatMessage(BaseModel):
    """채팅 메시지 모델 - 커스텀 검증"""
    role: str = Field(description="메시지 역할")
    content: str = Field(min_length=1, description="메시지 내용")

    @field_validator('role')
    @classmethod
    def validate_role(cls, v):
        """role은 user, assistant, system 중 하나여야 함"""
        allowed_roles = ["user", "assistant", "system"]
        if v not in allowed_roles:
            raise ValueError(f"role은 {allowed_roles} 중 하나여야 합니다.")
        return v
    
class ChatRequest(BaseModel):
    """채팅 요청 모델"""
    messages: List[ChatMessage] = Field(
        min_length=1,
        description="대화 메시지 목록"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="창의성 조절(0~2)"
    )
    max_tokens: int = Field(
        default=1000,
        ge=1,
        le=4096,
        description="최대 토큰 수"
    )

    # 예시 데이터 (Swagger UI에서 자동으로 표시됨)
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "messages": [
                        {"role": "system", "content": "당신은 친절한 AI 어시스턴트입니다."},
                        {"role": "user", "content": "안녕하세요!"}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
            ]
        }
    }

@app.post("/chat")
def chat(request: ChatRequest):
    """
    채팅 API(모델 연동 전 구조만)

    !!! 실제 LLM 연동 전에 API구조를 먼저 설계하는것이 좋다 !!!
    """
    return {
        "message": "채팅 요청 접수",
        "received": {
            "message_count": len(request.messages),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "last_message": request.messages[-1].content
        }
    }

# 4단계 : 응답 모델로 필터링
class UserInternal(BaseModel):
    """내부용 사용자 모델(민감정보 포함)"""
    id: int
    username: str
    email: str
    password_hash: str      # 민감정보!
    is_admin: bool

class UserPublic(BaseModel):
    """외부 노출용 사용자 모델"""
    id: int
    username: str
    # email, password_hash, is_admin 필드는 제외

# 가짜 유저 데이터(test용)
internal_users = [
    UserInternal(
        id=1,
        username="admin",
        email="admin@test.com",
        password_hash="hashed_secret_123",
        is_admin=True
    ),
    UserInternal(
        id=2,
        username="user1",
        email="user1@test.com",
        password_hash="hashed_password_456",
        is_admin=False
    )
]

@app.get("/users/{user_is}/public", response_model=UserPublic)
def get_public_user(user_is: int):
    """
    사용자 공개 정보 조회
    
    !!! response_model을 지정하면 해당 필드만 응답에 포함! - 민감정보가 실수로 노출되는 것을 방지합니다.
    """
    for user in internal_users:
        if user.id == user_is:
            return user
        
    raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")

@app.get("/users/{user_id}/internal", response_model=UserInternal)
def get_user_internal(user_id: int):
    """
    사용자 내부 정보 조회 (관리자용 = 실제로는 인증 필요하다)
    """
    for user in internal_users:
        if user.id == user_id:
            return user
        
    raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")