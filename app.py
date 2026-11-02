import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

# ページ全体のレイアウト設定（ワイド）
st.set_page_config(page_title="Haze Detection App", layout="wide")

# --- 1. タイトルと画面の完全分離（独立した上部エリア） ---
st.markdown("## 🌫️ ヘイズ検証用 Web アプリ（YOLO 物体検知）")
st.markdown("ファインチューニング済みモデル (`best.pt`) による物体検知および詳細データ検証デモ")
st.markdown("---")

# --- 2. 画面左部（サイドバー）での操作パネル ---
st.sidebar.header("⚙️ 設定パネル")

# モデルパスの指定
model_path = st.sidebar.text_input("モデルパス", value="best.pt")

# --- 信頼度閾値の同期管理 (スライダー ＆ 自由な数値入力) ---
if "conf_val" not in st.session_state:
    st.session_state.conf_val = 0.25

def update_from_slider():
    st.session_state.conf_val = st.session_state.slider_input

def update_from_num():
    st.session_state.conf_val = st.session_state.num_input

st.sidebar.markdown("### 🎯 信頼度閾値 (Confidence)")
# ① スライダーによる操作
conf_slider = st.sidebar.slider(
    "スライダー調整", 
    min_value=0.0, 
    max_value=1.0, 
    value=st.session_state.conf_val, 
    step=0.01,
    format="%.2f",
    key="slider_input",
    on_change=update_from_slider
)

# ② 好きな数値を直接入力できる欄
conf_num = st.sidebar.number_input(
    "数値直接入力", 
    min_value=0.0, 
    max_value=1.0, 
    value=st.session_state.conf_val, 
    step=0.01,
    format="%.2f",
    key="num_input",
    on_change=update_from_num
)

# 最終的な閾値として反映
conf_threshold = st.session_state.conf_val

st.sidebar.markdown("---")

# 複数画像のアップロード
uploaded_files = st.sidebar.file_uploader(
    "評価対象の画像を選択してください（複数選択可）", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

# 推論開始ボタン
start_inference = st.sidebar.button("🚀 推論開始", use_container_width=True)

# --- モデルのキャッシュ読み込み ---
@st.cache_resource
def load_model(path):
    try:
        return YOLO(path)
    except Exception as e:
        return None

model = load_model(model_path)

# --- セッションステートの初期化 ---
if "inference_done" not in st.session_state:
    st.session_state.inference_done = False
    st.session_state.results_cache = []

if "current_page" not in st.session_state:
    st.session_state.current_page = 1

# --- 推論ボタンが押されたときの処理 ---
if start_inference:
    if not uploaded_files:
        st.sidebar.warning("⚠️ 画像ファイルが選択されていません。")
        st.session_state.inference_done = False
    elif model is None:
        st.sidebar.error(f"❌ モデル '{model_path}' の読み込みに失敗しました。")
        st.
