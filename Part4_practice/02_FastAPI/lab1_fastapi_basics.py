from fastapi import FastAPI, HTTPException

# FastAPI 앱 인스턴스 생성
app = FastAPI(
    title="첫번째 FastAPI",
    description="AI 엔지니어 FastAPI 실습 1",
    version="1.0.0"
)

# 1단계: 가장 기본적인 GET 요청
@app.get("/")
def home():
    """루트 경로 - 서버가 잘 동작하는지 확인"""
    return {"message": "FastAPI 서버입니다!"}

@app.get("/health")
def health_check():
    """헬스체크 - 서버 상태 확인요 (실무에서 필수!)"""
    return {"status": "healthy"}

# 2단계: 경로 매개변수 (Path Parameter)
# URL 경로에 변수를 포함시키는 방식 
@app.get("/users/{user_id}")
def get_user(user_id: int):
    return {
        "user_id": user_id,
        "message": f"{user_id}번 사용자 정보입니다"
    }

@app.get("/items/{item_name}")
def get_item(item_name: str):
    return {
        "item_name": item_name,
        "message": f"'{item_name}' 아이템을 조회합니다."
    }

# 3단계: 쿼리 매개변수 (Query Parameter)
@app.get("/search")
def search_items(
    keyword: str,
    limit: int = 10,
    skip: int = 0,
):
    """
    쿼리 매개변수 예시
    - /search?keyword=AI → keyword="AI", limit=10, skip=0
    """
    return {
        "keyword": keyword,
        "limit": limit,
        "skip": skip,
        "message": f"{keyword}로 검색, {skip}번째부터 {limit}개 조회"
    }

# 4단계: 경로 + 쿼리 매개변수 조합
@app.get("/categories/{category}/products")
def get_products_by_category(
    category: str,
    min_price: int = 0,
    max_price: int = 100000,
    sort_by: str = "name"
):
    """
    실전 예시 : 카테고리별 상품 조회
    - /categories/electronics/products?min_price=1000&sort_by=price
    """
    return {
        "category": category,
        "filters": {
            "min_price": min_price,
            "max_price": max_price,
            "sort_by": sort_by
        },
        "message": f"{category} 카테고리 상품 조회"
    }

# 혼자해보기 1
@app.get("/greeting/{name}")
def greeting(name: str):
    return {"message": f"안녕하세요, {name}님!"}

@app.get("/calculate")
def calculate(a: int, b: int, operation: str = "add"):
    if operation == "add":
        result = a + b
    elif operation == "multiply":
        result = a * b
    else:
        raise HTTPException(status_code=400, detail="operation은 'add' 또는 'multiply'만 허용됩니다.")
    return {"a": a, "b": b, "operation": operation, "result": result}

@app.get("/movies/{genre}/list")
def list_movies(genre: str, year: int | None = None, rating: float = 0.0):
    filters = {"year": year, "rating": rating}
    return {
        "genre": genre,
        "filters": filters,
        "message": f"{genre} 장르 영화 리스트를 조회합니다."
    }