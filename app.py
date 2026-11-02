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
        st.session_state.inference_done = False
    else:
        with st.spinner("🔄 推論を実行中..."):
            cache = []
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file).convert("RGB")
                img_array = np.array(image)
                
                # YOLO推論の実行
                preds = model.predict(source=img_array, conf=conf_threshold, verbose=False)
                res = preds[0] if len(preds) > 0 else None
                
                cache.append({
                    "filename": uploaded_file.name,
                    "image": image,
                    "result": res
                })
            
            st.session_state.results_cache = cache
            st.session_state.inference_done = True
            st.session_state.current_page = 1  # 1ページ目にリセット

# --- 3. メイン画面の描画（結果表示エリア：タイトルとは完全に分離） ---
if st.session_state.inference_done and len(st.session_state.results_cache) > 0:
    cache = st.session_state.results_cache
    total_pages = len(cache)
    
    # ページ番号の範囲ガード
    if st.session_state.current_page < 1:
        st.session_state.current_page = 1
    elif st.session_state.current_page > total_pages:
        st.session_state.current_page = total_pages
        
    current_idx = st.session_state.current_page - 1
    current_data = cache[current_idx]
    
    filename = current_data["filename"]
    image = current_data["image"]
    res = current_data["result"]

    # 左右並列表示（元画像 ＆ 推論結果）
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"元画像（入力）: {filename}")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("推論結果（検出）")
        if res is not None:
            res_plotted = res.plot()  # BGR配列
            res_image = Image.fromarray(res_plotted[..., ::-1])  # RGB変換
            st.image(res_image, use_container_width=True)
        else:
            st.image(image, use_container_width=True)

    # --- 検出オブジェクト詳細データの表示 ---
    st.markdown("---")
    st.subheader(f"📊 検出オブジェクト詳細ログ: {filename}")

    if res is not None and len(res.boxes) > 0:
        det_data = []
        for box in res.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = res.names[cls_id]
            conf = float(box.conf[0].item())
            
            # 正規化座標の取得 [x_center, y_center, width, height]
            x_center, y_center, w, h = box.xywhn[0].tolist()
            
            det_data.append({
                "class": cls_name,
                "conf": f"{conf:.4f}",
                "BB Coordinate (x_ctr, y_ctr, w, h)": f"({x_center:.3f}, {y_center:.3f}, {w:.3f}, {h:.3f})"
            })

        df_det = pd.DataFrame(det_data)
        st.dataframe(df_det, use_container_width=True)
        st.success(f"検知されたオブジェクト総数: {len(res.boxes)} 個 (信頼度閾値: {conf_threshold:.2f})")
    else:
        st.info("ℹ️ 設定された条件を満たすオブジェクトは検出されませんでした。")

    # --- 4. メイン画面下部のページネーション（自由な数値入力と移動ボタン） ---
    st.markdown("---")
    
    p_col1, p_col2, p_col3, p_col4 = st.columns([1, 1, 2, 1])
    
    with p_col1:
        if st.button("◀ 前の画像", use_container_width=True, disabled=(st.session_state.current_page <= 1)):
            st.session_state.current_page -= 1
            st.rerun()
            
    with p_col2:
        if st.button("次の画像 ▶", use_container_width=True, disabled=(st.session_state.current_page >= total_pages)):
            st.session_state.current_page += 1
            st.rerun()
            
    with p_col3:
        new_page = st.number_input(
            f"ページ移動 (全 {total_pages} 件)", 
            min_value=1, 
            max_value=total_pages, 
            value=st.session_state.current_page,
            step=1
        )
        if new_page != st.session_state.current_page:
            st.session_state.current_page = new_page
            st.rerun()
            
    with p_col4:
        st.markdown(f"<br><div style='text-align: center;'><b>{st.session_state.current_page} / {total_pages}</b></div>", unsafe_allow_html=True)

else:
    if not st.session_state.inference_done:
        st.info("👈 左側のサイドバーから画像を選択し、**「🚀 推論開始」**ボタンを押してください。")
