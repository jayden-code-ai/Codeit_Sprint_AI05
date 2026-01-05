import streamlit as st
import pandas as pd
import pydeck as pdk

st.title("PyDeck Chart 기본예제")

# 예제 테이터 (위도 경도)
df = pd.DataFrame({
"lat": [37.5665, 37.5651, 37.5640],
"lon": [126.9780, 126.9895, 126.9750]
})
# PyDeck 레이어 정의
layer = pdk.Layer(
"ScatterplotLayer",
data=df,
get_position="[lon, lat]",
get_radius=100,
get_color=[255, 0, 0],
pickable=True
)
# 지도뷰 설정
view_state = pdk.ViewState(
latitude=37.5665,
longitude=126.9780,
zoom=12
)
# 차트 생성
deck = pdk.Deck(
layers=[layer],
initial_view_state=view_state
)
# Streamlit에 출력
st.pydeck_chart(deck)