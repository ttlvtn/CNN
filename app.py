import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 設置頁面標題
st.set_page_config(page_title="CNN 核心理論簡報互動工具", layout="wide")

st.title("第 4 章：卷積神經網絡 (CNN) 核心機制簡報")
st.write("本工具根據《深度學習详解》來源製作，用於展示卷積層的設計邏輯與參數影響。")

# 側邊欄：參數調整區
st.sidebar.header("🔧 卷積層超參數調整")
input_size = st.sidebar.slider("原始圖像尺寸 (Width/Height)", 5, 100, 10) # 來源預設為 100x100 [1]
kernel_size = st.sidebar.slider("濾波器尺寸 (Kernel Size)", 1, 11, 3, step=2) # 來源提到 3x3 最常見 [2]
stride = st.sidebar.slider("步幅 (Stride)", 1, 5, 1) # 來源提到通常設為 1 或 2 [3]
padding = st.sidebar.slider("零填充 (Padding)", 0, 5, 0) # 來源提到超出範圍就補 0 [3]

# 理論計算：輸出尺寸公式
# 根據來源圖 4.20 的邏輯，計算掃描後的特徵映射大小
output_size = int((input_size - kernel_size + 2 * padding) / stride) + 1

# 1. 圖像輸入展示
st.header("1. 圖像作為模型輸入")
col1, col2 = st.columns(2)
with col1:
    st.write("### 3D 張量表示")
    st.write(f"彩色圖像包含：**寬 x 高 x 通道 (Channel)** [4]")
    st.write(f"目前的輸入設定：`{input_size} x {input_size} x 3` (RGB)")
    st.info("來源指出：彩色圖像的像素由紅、綠、藍組成，這 3 種顏色稱為色彩通道 [4]。")
with col2:
    st.write("### 全連接層的挑戰")
    param_count = (input_size * input_size * 3) * 1000
    st.write(f"若第一層有 1000 個神經元，權重數量將高達：**{param_count:,}**")
    st.warning("過多的參數會增加模型彈性，但也大大增加了『過擬合 (Overfitting)』的風險 [1]。")

st.divider()

# 2. 卷積核心機制
st.header("2. 卷積層的簡化邏輯")
tab1, tab2, tab3 = st.tabs(["感受野 (Receptive Field)", "參數共享 (Parameter Sharing)", "特徵映射 (Feature Map)"])

with tab1:
    st.write("### 觀察 1：檢測模式不需要整張圖像")
    st.write(f"神經元只需要看一小部分範圍（即**感受野**）就能檢測關鍵模式（如鳥嘴、眼睛）[5]。")
    st.write(f"目前設定的感受野大小：**{kernel_size} x {kernel_size}** [2]")
    
    # 簡易畫布模擬感受野掃描
    fig, ax = plt.subplots(figsize=(4, 4))
    grid = np.zeros((input_size, input_size))
    grid[0:kernel_size, 0:kernel_size] = 1 # 標示第一個感受野
    ax.imshow(grid, cmap='Blues')
    ax.set_title("藍色區域即為目前的感受野範圍")
    st.pyplot(fig)

with tab2:
    st.write("### 觀察 2：同樣的模式會出現在不同區域")
    st.write("既然不同位置的模式（如鳥嘴）功能相同，沒必要為每個位置放獨立的檢測器 [6, 7]。")
    st.success(f"**參數共享**：不同感受野的神經元共用同一組權重，這組共用的參數稱為**濾波器 (Filter)** [8, 9]。")

with tab3:
    st.write("### 輸出結果：特徵映射")
    st.write(f"當濾波器掃過整張圖像進行內積運算後，會產生一組新的數字 [10, 11]。")
    if output_size > 0:
        st.metric("產出的特徵映射尺寸 (Output Size)", f"{output_size} x {output_size}")
    else:
        st.error("目前的參數設定導致輸出尺寸小於 0，請調小核大小或步幅。")

st.divider()

# 3. 匯聚層 (Pooling)
st.header("3. 下採樣與匯聚")
pooling_type = st.radio("匯聚方式", ["最大匯聚 (Max Pooling)", "平均匯聚 (Mean Pooling)"], horizontal=True)
st.write("### 觀察 3：下採樣不影響模式檢測")
st.write("將鳥的圖像縮小，它還是一隻鳥。匯聚操作可以減少運算量，且沒有需要學習的參數 [12, 13]。")
if pooling_type == "最大匯聚 (Max Pooling)":
    st.info("最大匯聚：在每一組數字中選出最大的一個作為代表 [13]。")
else:
    st.info("平均匯聚：取每一組數字的平均值作為代表 [13]。")

st.divider()

# 4. 特殊應用案例：AlphaGo
st.header("4. 卷積神經網絡的特殊應用：下圍棋")
col_go1, col_go2 = st.columns(2)
with col_go1:
    st.write("### 棋盤即圖像")
    st.write("- 棋盤解析度：**19 x 19** [14]")
    st.write("- AlphaGo 輸入通道：**48 個通道**（由圍棋高手設計）[14]")
    st.write("- 關鍵發現：棋盤上的吃子模式只需要看 5x5 的小範圍即可 [15]")
with col_go2:
    st.write("### AlphaGo 的獨特設計")
    st.error("**不使用匯聚層 (No Pooling)**")
    st.write("原因：下圍棋是精細任務，隨便拿掉一行一列都會影響判斷 [15]。")
    st.write("AlphaGo 第一層濾波器大小為 **5x5**，步幅為 **1**，並使用 **ReLU** [16]。")

st.divider()
st.write("💡 **簡報總結**：CNN 透過感受野、參數共享與匯聚，專門為圖像特性設計，能有效處理模式偵測並降低過擬合風險 [17]。")
