import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Haze Detection App", layout="wide")
st.title("🌫️ ヘイズ検証用 Web アプリ（YOLO 物体検知）")
st.write("ファインチューニング済みモデル (best.pt) による物体検知デモ")

# モデルパスの指定
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

# --- 複数画像のアップロード ---
uploaded_files = st.file_uploader("評価対象の画像を選択してください（複数選択可）", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files and model is not None:
    # --- 推論開始ボタン ---
    if st.button("🚀 推論開始"):
        st.write("---")
        
        # 複数枚の画像をループ処理で1枚ずつ推論・表示する
        for uploaded_file in uploaded_files:
            # 画像の読み込み (RGB形式)
            image = Image.open(uploaded_file).convert("RGB")
            
            # YOLOによる推論実行（NumPy配列にせず、Pillowの画像を直接渡すことで色の狂いを防ぎます）
            results = model.predict(source=image, conf=conf_threshold, verbose=False)
            
            # 推論結果の画像を取得 (bgr=False を指定して、枠線の色等も含めてRGBで取得)
            res_plotted = results[0].plot(bgr=False)
            
            # 画面への表示レイアウト（左右2列）
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"入力画像: {uploaded_file.name}")
                st.image(image, use_container_width=True)
                
            with col2:
                st.subheader("物体検知結果")
                # channels="BGR" の指定を削除し、そのまま（デフォルトのRGBとして）表示
                st.image(res_plotted, use_container_width=True)
                
            # 検出数の表示
            boxes = results[0].boxes
            st.success(f"[{uploaded_file.name}] 検知された物体数: {len(boxes)} 個 (信頼度閾値: {conf_threshold})")
            st.write("---")
