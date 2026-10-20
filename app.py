import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Haze Detection App", layout="wide")
st.title("🌫️ ヘイズ検証用 Web アプリ（YOLO 物体検知）")
st.write("ファインチューニング済みモデル (best.pt) による物体検知デモ")

# モデルのロード (サイドバーでパス調整可能)
model_path = st.sidebar.text_input("モデルパス", value="best.pt")

@st.cache_resource
def load_model(path):
    try:
        return YOLO(path)
    except Exception as e:
        st.error(f"モデルの読み込みに失敗しました: {e}")
        return None

model = load_model(model_path)

# 信頼度閾値のスライダー
conf_threshold = st.slider("信頼度閾値 (Confidence Threshold)", 0.0, 1.0, 0.25, 0.05)

uploaded_file = st.file_uploader("評価対象の画像を選択してください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # 画像の読み込み
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    # YOLOによる推論実行
    results = model.predict(source=img_array, conf=conf_threshold, verbose=False)
    
    # 推論結果のバウンディングボックスを描画した画像を取得
    res_plotted = results[0].plot()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("入力画像")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("物体検知結果")
        st.image(res_plotted, channels="BGR", use_container_width=True)
        
    # 検出数の表示
    boxes = results[0].boxes
    st.success(f"検知された物体数: {len(boxes)} 個 (信頼度閾値: {conf_threshold})")
