import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# --- 頁面配置 ---
st.set_page_config(page_title="李宏毅 AI 教室：CNN 參數視覺化", layout="wide")

# --- 載入預訓練模型 (MobileNet V2) ---
@st.cache_resource
def load_cnn_model():
    # 載入分類模型
    model = hub.load("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4")
    # 載入標籤
    labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    with open(labels_path) as f:
        labels = f.read().splitlines()
    return model, labels

model, imagenet_labels = load_cnn_model()

# --- 主介面 ---
st.title("🖼️ 第一種應用：影像辨識與 CNN 理論參數實驗室")
st.markdown("""
本區塊結合了 **影像種類判斷** 與 **李宏毅老師《深度學習详解》第四章** 的核心參數。
你可以觀察改變『感受野』或『池化』如何讓 AI 變聰明或變笨。
""")

# --- 側邊欄：參數調整區 ---
st.sidebar.header("🛠️ CNN 理論參數設定")
k_size = st.sidebar.slider("1. 感受野大小 (Kernel Size)", 1, 11, 3, step=2, help="對應神經元觀察局部特徵的範圍")
k_stride = st.sidebar.slider("2. 步幅 (Stride)", 1, 5, 1, help="滑動的距離，越大則輸出的特徵圖越小")
use_pooling = st.sidebar.checkbox("3. 啟用池化層 (Max Pooling)", value=False, help="模擬特徵壓縮，增加平移不變性")
if use_pooling:
    p_size = st.sidebar.slider("池化視窗大小", 2, 4, 2)

# --- 上傳與辨識區 ---
up_file = st.file_uploader("請上傳一張圖片（如貓、狗、車、水果等）", type=['jpg', 'jpeg', 'png'])

if up_file:
    # 1. 影像預處理
    raw_img = Image.open(up_file).convert('RGB')
    display_img = raw_img.resize((224, 224))
    img_tensor = tf.convert_to_tensor(np.array(display_img, dtype=np.float32)/255.0)[tf.newaxis, ...]

    # 2. 進行種類辨識
    logits = model(img_tensor)
    probs = tf.nn.softmax(logits).numpy()
    top_idx = np.argsort(probs[0])[-5:][::-1] # 取前五名

    # --- 畫面佈局 ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("✅ AI 辨識結果")
        st.image(display_img, caption="AI 輸入影像 (224x224)", use_container_width=True)
        
        # 顯示前三名機率
        st.write("🔍 **預測前三名：**")
        for i in range(3):
            label = imagenet_labels[top_idx[i]]
            score = probs[0][top_idx[i]]
            st.info(f"排名 {i+1}: **{label}** ({score:.2%})")

    with col2:
        st.subheader("🔬 理論視覺化：特徵提取過程")
        
        # 模擬卷積運算 (使用 OpenCV 根據參數模擬)
        gray_img = cv2.cvtColor(np.array(display_img), cv2.COLOR_RGB2GRAY)
        
        # 根據參數模擬特徵提取 (Filter 為邊緣偵測)
        kernel = np.ones((k_size, k_size), np.float32) * -1
        kernel[k_size//2, k_size//2] = (k_size**2) - 1
        
        # 執行卷積
        feat_map = cv2.filter2D(gray_img, -1, kernel)
        
        # 模擬步幅 (Stride) - 透過降採樣實現
        feat_map = feat_map[::k_stride, ::k_stride]
        
        # 模擬池化 (Max Pooling)
        if use_pooling:
            feat_map = cv2.dilate(feat_map, np.ones((p_size, p_size), np.uint8))
            feat_map = feat_map[::p_size, ::p_size]

        st.image(feat_map, caption=f"特徵圖 (Feature Map) - 當前尺寸: {feat_map.shape}", clamp=True, use_container_width=True)
        st.caption(f"目前的特徵圖展現了 AI 在 Kernel={k_size} 下捕捉到的邊緣線條。")

    # --- 理論解釋區 ---
    st.divider()
    st.subheader("📘 參數影響實驗紀錄表")
    
    exp_col1, exp_col2, exp_col3 = st.columns(3)
    
    with exp_col1:
        st.write("🔎 **當 Kernel Size 變大時：**")
        st.write("- **視覺效果**：捕捉到的邊緣會變粗，特徵變模糊。")
        st.write("- **理論連結**：神經元的**感受野**變大，能看到更大塊的物件（如眼睛），但會失去微小的線條。")
        

    with exp_col2:
        st.write("🏃 **當 Stride 變大時：**")
        st.write("- **視覺效果**：特徵圖變得非常小且破碎。")
        st.write("- **理論連結**：這是**下採樣**的一種。李老師提到，為了減少運算量，我們會跳著掃描，但代價是遺漏細節。")
        

    with exp_col3:
        st.write("🧊 **當啟用池化 (Pooling) 時：**")
        st.write("- **視覺效果**：特徵圖被大幅度壓縮，但保留了最亮的特徵點。")
        st.write("- **理論連結**：**平移不變性**。即便貓在圖片左邊或右邊，池化後的強特徵是一樣的。")
        

else:
    st.warning("請先上傳一張圖片以開始實驗。")

# --- 頁尾 ---
st.sidebar.markdown("---")
st.sidebar.caption("💡 提示：試著把 Stride 調到最大，你會發現特徵圖變得無法辨識，這就是為什麼步幅不能太大的原因。")
