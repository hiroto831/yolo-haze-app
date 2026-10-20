import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance

st.set_page_config(page_title="Haze Detection App", layout="wide")
st.title("🌫️ ヘイズ検証用 Web アプリ")
st.write("画像アップロードおよびヘイズ視認性テスト")

uploaded_file = st.file_uploader("評価対象の画像を選択してください", type=["jpg", "jpeg", "png"])

conf_threshold = st.slider("信頼度閾値 (Confidence Threshold)", 0.0, 1.0, 0.25, 0.05)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("入力画像")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("処理結果 (Preview)")
        # 画像プレビュー表示（Pillowによる安全な処理）
        st.image(image, use_container_width=True)
        st.success(f"画像の読み込みに成功しました。(閾値: {conf_threshold})")
