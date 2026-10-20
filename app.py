import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Haze Detection App", layout="wide")
st.title("🌫️ ヘイズ検証用 Web アプリ（YOLO 物体検知）")
st.write("ファインチューニング済みモデル (best.pt) による物体検知および詳細データ検証デモ")

# モデルパスの指定
model_path = st.sidebar.text_input("モデルパス", value="models/best.pt")

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
        
        # 複数枚の画像をループ処理
        for uploaded_file in uploaded_files:
            # 1. 画像の読み込み
            image = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(image)
            
            # 2. YOLOによる推論実行
            results = model.predict(source=img_array, conf=conf_threshold, verbose=False)[0]
            
            # 3. 描画画像の作成（BGRからRGBへ安全に変換）
            res_plotted = results.plot()  # BGRのnumpy配列
            res_image = Image.fromarray(res_plotted[..., ::-1])  # RGBに変換
            
            # 4. 左右並列表示
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"元画像（入力）: {uploaded_file.name}")
                st.image(image, use_container_width=True)
            with col2:
                st.subheader("推論結果（検出）")
                st.image(res_image, use_container_width=True)

            # 5. --- 検出データの詳細表示 ---
            st.markdown("---")
            st.subheader(f"📊 検出オブジェクト詳細データ: {uploaded_file.name}")

            boxes = results.boxes
            if len(boxes) > 0:
                det_data = []
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = results.names[cls_id]
                    conf = float(box.conf[0].item())
                    
                    # 正規化座標の取得 [x_center, y_center, width, height]
                    x_center, y_center, w, h = box.xywhn[0].tolist()
                    
                    det_data.append({
                        "クラス名": cls_name,
                        "信頼度 (Confidence)": f"{conf:.4f}",
                        "中心X (割合)": f"{x_center:.4f}",
                        "中心Y (割合)": f"{y_center:.4f}",
                        "幅 (割合)": f"{w:.4f}",
                        "高さ (割合)": f"{h:.4f}"
                    })

                df_det = pd.DataFrame(det_data)
                st.dataframe(df_det, use_container_width=True)
                st.success(f"検知された総オブジェクト数: {len(boxes)} 個")
            else:
                st.info("設定された条件を満たすオブジェクトは検出されませんでした。サイドバーの信頼度閾値を下げてみてください。")
            
            st.write("====================")
