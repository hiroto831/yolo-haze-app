import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import os

# --- ページ基本設定 ---
st.set_page_config(
    page_title="YOLO26 ヘイズ画像検出確認実験",
    layout="wide"
)

st.title("🚗 ヘイズ（もや）環境下におけるYOLO26モデル確認実験Webアプリ")
st.write("ファインチューニング済みモデル (`best.pt`) を用いて、未知のヘイズ画像に対する検出精度をリアルタイムに検証します。")

# --- サイドバー：検出パラメータ設定 ---
st.sidebar.header("🛠️ 検出パラメータ設定")

# 1. モデルの読み込み (キャッシュ化して高速化)
@st.cache_resource
def load_yolo_model(model_path):
    return YOLO(model_path)

model_path = "models/best.pt"
if os.path.exists(model_path):
    model = load_yolo_model(model_path)
    st.sidebar.success("`best.pt` の読み込みに成功しました")
else:
    st.sidebar.error(f"モデルファイルが見つかりません: {model_path}")
    st.stop()

# 2. 信頼度 (Confidence) 閾値スライダー
conf_threshold = st.sidebar.slider(
    "信頼度閾値 (Confidence Threshold)", 
    min_value=0.0, max_value=1.0, value=0.25, step=0.05,
    help="設定した信頼度以上のバウンディングボックスのみを表示します。"
)

# 3. IoU (NMS) 閾値スライダー
iou_threshold = st.sidebar.slider(
    "IoU 閾値 (NMS Threshold)", 
    min_value=0.0, max_value=1.0, value=0.45, step=0.05,
    help="重なるバウンディングボックスを整理するNMSの計算閾値です。"
)

# 4. 検出対象クラスの選択
selected_classes = st.sidebar.multiselect(
    "検出対象クラス",
    options=["person", "car"],
    default=["person", "car"]
)

# クラスIDマッピング (person=0, car=1)
class_map = {"person": 0, "car": 1}
target_class_ids = [class_map[c] for c in selected_classes if c in class_map]

# --- メインコンテンツ：推論実験 ---
st.header("📸 テスト画像の読み込みと推論実行")
uploaded_file = st.file_uploader("確認テスト用ヘイズ画像をアップロードしてください", type=["jpg", "jpeg", "png"])

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

    # 推論結果画像の生成
    res_plotted = results.plot()  # BGR形式
    res_image = Image.fromarray(res_plotted[..., ::-1])  # RGB変換

    # 左右2画面で並列比較
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("元画像（入力）")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("推論結果（検出）")
        st.image(res_image, use_container_width=True)

    # 検出されたオブジェクトの詳細データ表示
    st.markdown("---")
    st.subheader("📊 検出オブジェクト詳細データ")

    boxes = results.boxes
    if len(boxes) > 0:
        det_data = []
        for box in boxes:
            cls_id = int(box.cls[0].item())
            cls_name = results.names[cls_id]
            conf = float(box.conf[0].item())
            
            # 正規化座標 (x_center, y_center, width, height)
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
        st.info("条件を満たすオブジェクトは検出されませんでした。信頼度閾値を下げてみてください。")

# --- 研究評価資料タブ ---
st.markdown("---")
st.header("📈 モデル学習結果・評価データ（参考資料）")
tab1, tab2 = st.tabs(["学習推移 (results.png)", "混同行列 (confusion_matrix.png)"])

with tab1:
    if os.path.exists("assets/results.png"):
        st.image("assets/results.png", caption="Loss推移および Precision, Recall, mAP50, mAP50-95 の変化", use_container_width=True)
    else:
        st.warning("assets/results.png が配置されていません。")

with tab2:
    if os.path.exists("assets/confusion_matrix.png"):
        st.image("assets/confusion_matrix.png", caption="混同行列（背景・クラス間の誤認傾向）", use_container_width=True)
    else:
        st.warning("assets/confusion_matrix.png が配置されていません。")