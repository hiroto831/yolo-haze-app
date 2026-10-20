import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import numpy as np

# ページ設定
st.set_page_config(
    page_title="YOLO26 ヘイズ画像物体検出・確認実験Webアプリ",
    layout="wide"
)

st.title("🚗 ヘイズ（もや）環境下におけるYOLO26モデル確認実験Webアプリ")
st.write("ファインチューニング済みモデル (`best.pt`) を用いて、未知のヘイズ画像に対する検出精度をインタラクティブに検証します。")

# --- サイドバー：モデルとパラメータの設定 ---
st.sidebar.header("🛠️ 検出パラメータ設定")

# 1. モデルのロード
@st.cache_resource
def load_yolo_model(model_path):
    return YOLO(model_path)

try:
    model = load_yolo_model("models/best.pt")
    st.sidebar.success("`best.pt` の読み込みに成功しました")
except Exception as e:
    st.sidebar.error(f"モデルの読み込みエラー: {e}")
    st.stop()

# 2. 信頼度・IoU閾値スライダー
conf_threshold = st.sidebar.slider(
    "信頼度閾値 (Confidence Threshold)", 
    min_value=0.0, max_value=1.0, value=0.25, step=0.05,
    help="設定した信頼度以上のバウンディングボックスのみを表示します。"
)

iou_threshold = st.sidebar.slider(
    "IoU 閾値 (NMS Threshold)", 
    min_value=0.0, max_value=1.0, value=0.45, step=0.05,
    help="重なるバウンディングボックスを排除するNMSの計算閾値です。"
)

# 3. クラスフィルター
selected_classes = st.sidebar.multiselect(
    "検出対象クラス",
    options=["person", "car"],
    default=["person", "car"]
)

# クラスIDのマッピング (YOLO側設定に合わせて調整)
class_map = {"person": 0, "car": 1}
target_class_ids = [class_map[c] for c in selected_classes]

# --- メインコンテンツ：確認実験 ---
st.header("📸 テスト画像の読み込みと推論実行")
uploaded_file = st.file_uploader("テスト用ヘイズ画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # 推論の実行
    results = model.predict(
        source=image,
        conf=conf_threshold,
        iou=iou_threshold,
        classes=target_class_ids if target_class_ids else None,
        verbose=False
    )[0]

    # 結果画像の作成
    res_plotted = results.plot()  # BGRのnumpy配列
    res_image = Image.fromarray(res_plotted[..., ::-1])  # RGBに変換

    # 左右並列表示
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("元画像（入力）")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("推論結果（検出）")
        st.image(res_image, use_container_width=True)

    # --- 検出データの詳細表示 ---
    st.markdown("---")
    st.subheader("📊 検出オブジェクト詳細データ")

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
    else:
        st.info("設定された条件を満たすオブジェクトは検出されませんでした。サイドバーの信頼度閾値を下げてみてください。")

# --- 研究評価資料タブ ---
st.markdown("---")
st.header("📈 モデル学習結果・評価データ（参考資料）")
tab1, tab2 = st.tabs(["学習推移 (results.png)", "混同行列 (confusion_matrix.png)"])

with tab1:
    st.image("assets/results.png", caption="Loss推移および Precision, Recall, mAP50, mAP50-95 の変化", use_container_width=True)
with tab2:
    st.image("assets/confusion_matrix.png", caption="混同行列（背景・クラス間の誤認傾向）", use_container_width=True)