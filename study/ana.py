import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import timedelta

# 페이지 설정
st.set_page_config(page_title="비트코인 실시간 분석 대시보드", layout="wide", page_icon="₿")

@st.cache_data
def load_data(file_path):
    """
    로컬 CSV 파일을 불러와 전처리하는 함수
    """
    if not os.path.exists(file_path):
        return None

    # 데이터 로드 (세미콜론 구분자 및 인코딩 처리)
    df = pd.read_csv(file_path, sep=';')
    
    # 날짜 데이터 변환
    date_col = 'timestamp' if 'timestamp' in df.columns else 'timeOpen'
    df[date_col] = pd.to_datetime(df[date_col])
    df['date_only'] = df[date_col].dt.date
    
    # 시간순 정렬
    df = df.sort_values(date_col)
    return df

def predict_next_day(df):
    """
    선형회귀를 사용하여 내일의 가격을 예측하는 함수
    """
    # 학습을 위한 데이터 준비 (최근 30일 데이터 사용)
    df_recent = df.tail(30).copy()
    
    # 날짜를 숫자로 변환 (학습용)
    df_recent['date_ordinal'] = pd.to_datetime(df_recent['timestamp']).map(pd.Timestamp.toordinal)
    
    X = df_recent[['date_ordinal']].values
    y = df_recent['close'].values
    
    # 모델 학습
    model = LinearRegression()
    model.fit(X, y)
    
    # 내일 날짜 계산
    last_date = df_recent['timestamp'].max()
    next_date = last_date + timedelta(days=1)
    next_date_ordinal = np.array([[next_date.toordinal()]])
    
    # 예측
    prediction = model.predict(next_date_ordinal)[0]
    
    # 모델 성능 (R^2)
    score = model.score(X, y)
    
    return prediction, next_date, score

def main():
    st.title("₿ 비트코인(BTC) 로컬 데이터 분석기")
    
    csv_file = 'btcoin.csv'
    
    # 1. 데이터 불러오기
    df = load_data(csv_file)
    
    if df is None:
        st.error(f"파일을 찾을 수 없습니다: '{csv_file}' 파일이 파이썬 스크립트와 같은 폴더에 있는지 확인해주세요.")
        return

    # 2. 사이드바 필터 및 예측 결과
    st.sidebar.header("🔍 분석 설정")
    min_date = df['date_only'].min()
    max_date = df['date_only'].max()
    
    try:
        date_range = st.sidebar.date_input(
            "분석 기간 선택",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    except:
        date_range = [min_date, max_date]

    # 데이터 필터링
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = df[(df['date_only'] >= start_date) & (df['date_only'] <= end_date)].copy()
    else:
        filtered_df = df.copy()

    # --- 내일 가격 예측 섹션 (사이드바 하단) ---
    st.sidebar.markdown("---")
    st.sidebar.header("🔮 내일 가격 예측 (AI)")
    if len(df) > 5:
        pred_price, pred_date, model_score = predict_next_day(df)
        current_price = df.iloc[-1]['close']
        
        diff = pred_price - current_price
        diff_pct = (diff / current_price) * 100
        
        status = "상승 예상 ▲" if diff > 0 else "하락 예상 ▼"
        color = "green" if diff > 0 else "red"
        
        st.sidebar.subheader(f"{pred_date.strftime('%Y-%m-%d')}")
        st.sidebar.markdown(f"**결과: <span style='color:{color}'>{status}</span>**", unsafe_allow_html=True)
        st.sidebar.write(f"예측가: ₩{pred_price:,.0f}")
        st.sidebar.write(f"예상 변동: {diff_pct:+.2f}%")
        st.sidebar.caption(f"모델 신뢰도 (R²): {model_score:.2f}")
        st.sidebar.info("※ 선형회귀 모델 기반 예측으로 실제 투자 결과와 다를 수 있습니다.")
    else:
        st.sidebar.warning("데이터가 부족하여 예측을 진행할 수 없습니다.")

    # 3. 상단 주요 지표 (KPI)
    if not filtered_df.empty:
        latest = filtered_df.iloc[-1]
        prev = filtered_df.iloc[-2] if len(filtered_df) > 1 else latest
        
        change = latest['close'] - prev['close']
        change_pct = (change / prev['close']) * 100

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("현재 종가", f"₩{latest['close']:,.0f}", f"{change_pct:+.2f}%")
        m2.metric("기간 최고가", f"₩{filtered_df['high'].max():,.0f}")
        m3.metric("기간 최저가", f"₩{filtered_df['low'].min():,.0f}")
        m4.metric("거래 횟수", f"{len(filtered_df):,} 일")

    # 4. 메인 차트: 가격 추이
    st.subheader("📈 가격 변동 추이")
    chart_type = st.radio("차트 형태", ["라인 차트", "캔들스틱"], horizontal=True)

    if chart_type == "라인 차트":
        fig = px.line(filtered_df, x='timestamp', y='close', title="비트코인 종가 추이")
    else:
        fig = go.Figure(data=[go.Candlestick(
            x=filtered_df['timestamp'],
            open=filtered_df['open'],
            high=filtered_df['high'],
            low=filtered_df['low'],
            close=filtered_df['close'],
            name="BTC"
        )])
    
    fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # 5. 거래량 및 변동성 분석
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📊 거래량 분석")
        vol_fig = px.bar(filtered_df, x='timestamp', y='volume', color='volume', title="일일 거래량")
        st.plotly_chart(vol_fig, use_container_width=True)
        
    with col_right:
        st.subheader("📉 시가 대비 고가 변동성")
        filtered_df['volatility'] = (filtered_df['high'] - filtered_df['open']) / filtered_df['open'] * 100
        volat_fig = px.area(filtered_df, x='timestamp', y='volatility', title="일일 변동성 (%)", color_discrete_sequence=['red'])
        st.plotly_chart(volat_fig, use_container_width=True)

    # 6. 데이터 테이블
    with st.expander("📝 원본 데이터 상세보기"):
        st.write(f"총 {len(filtered_df)}개의 레코드가 조회되었습니다.")
        st.dataframe(filtered_df.sort_values('timestamp', ascending=False), use_container_width=True)

if __name__ == "__main__":
    main()