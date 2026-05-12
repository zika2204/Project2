import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os

# =========================================
# CẤU HÌNH GIAO DIỆN XANH TOÀN DIỆN
# =========================================
st.set_page_config(page_title="MOTO CŨ VN", page_icon="🏍️", layout="centered")

st.markdown("""
<style>
    /* 1. Tiêu đề MOTO CŨ VN Cực đại */
    .title { 
        text-align: center; 
        font-size: 100px; 
        font-weight: 900; 
        color: #1E88E5; 
        margin-top: -20px;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.1);
    }
    
    /* 2. Logo căn giữa */
    .logo-container { text-align: center; margin-top: 20px; }
    .logo-img { width: 180px; }

    /* 3. Chỉnh màu Xanh cho Slider (Thanh kéo) */
    .stSlider [data-baseweb="slider"] [aria-valuemax] {
        background-color: #1E88E5;
    }
    .stSlider [data-baseweb="slider"] [data-testid="stTickBar"] {
        display: none;
    }

    /* 4. Chỉnh màu Xanh cho Radio Button (Nút chọn) */
    div[data-testid="stRadio"] > div {
        background-color: #f0f7ff;
        padding: 10px;
        border-radius: 10px;
    }
    /* Chỉnh màu chấm tròn khi chọn */
    div[data-testid="stRadio"] label div[data-testid="stMarkdownContainer"] p {
        color: #0D47A1;
        font-weight: bold;
    }

    /* 5. Nút bấm định giá to và xanh */
    div.stButton > button {
        background-color: #1E88E5 !important;
        color: white !important;
        border-radius: 20px !important;
        height: 60px !important;
        font-size: 22px !important;
        border: none !important;
    }

    /* 6. Khung kết quả */
    .result-box { 
        background-color: #E3F2FD; 
        padding: 35px; 
        border-radius: 25px; 
        border: 4px solid #1E88E5; 
        text-align: center; 
    }
    .price-text { color: #0D47A1; font-size: 55px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# =========================================
# XỬ LÝ LOGO (HIỂN THỊ TỪ GITHUB HOẶC LOCAL)
# =========================================
# Thay link này bằng link file logo của bạn trên GitHub (dạng Raw)
# Hoặc nếu chạy máy tính thì để "logo.png"
logo_url = "logoP2.png" 

st.markdown(f"""
    <div class="logo-container">
        <img src="{logo_url}" class="logo-img">
        <p class="title">MOTO CŨ VN</p>
    </div>
""", unsafe_allow_html=True)

# =========================================
# LOAD DATA & ML (Giữ nguyên bộ lọc sạch)
# =========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("xecu.csv")
        df.columns = df.columns.str.strip().str.lower()
        
        def clean(v):
            c = "".join(filter(str.isdigit, str(v)))
            return float(c) if c else 0.0

        df["price_numeric"] = df["price"].apply(clean)
        df["odo_numeric"] = df["odo"].apply(clean)
        df = df[df["price_numeric"] > 1000000]
        df["is_repaired"] = df["repaired_parts"].astype(str).str.lower().str.contains("yes|có").astype(int)
        
        brands = sorted(df["brand"].unique())
        models = sorted(df["model"].unique())
        locs = sorted(df["location"].unique())
        df_ml = pd.get_dummies(df, columns=["brand", "model", "location"])
        return df_ml, brands, models, locs
    except:
        return None, None, None, None

df_ml, brands, models, locations = load_data()

# =========================================
# GIAO DIỆN NHẬP LIỆU
# =========================================
if df_ml is not None:
    c1, c2 = st.columns(2)
    with c1:
        u_brand = st.selectbox("Hãng xe", brands)
        u_year = st.number_input("Năm sản xuất", 2010, 2026, 2023)
        # THANH KÉO (Màu xanh đã được chỉnh bằng CSS phía trên)
        u_cond = st.slider("Độ mới xe (1-10)", 1, 10, 8)
    with c2:
        u_model = st.selectbox("Dòng xe", models)
        u_odo = st.number_input("Số KM đã đi", 0, 300000, 5000)
        # NÚT CHỌN (Màu xanh đã được chỉnh bằng CSS phía trên)
        u_rep = st.radio("Tình trạng phụ tùng", ["Còn Zin (Chưa thay)", "Đã sửa chữa/Thay thế"])
        u_loc = st.selectbox("Khu vực", locations)

    # ML Train
    f_cols = [c for c in df_ml.columns if any(x in c for x in ["brand_", "model_", "location_"])]
    f_cols += ["year", "odo_numeric", "condition ", "is_repaired"]
    # Lưu ý: check lại tên cột 'condition ' có dấu cách ở cuối hay không từ file csv
    
    X = df_ml[f_cols]
    y = df_ml["price_numeric"]
    model = LinearRegression().fit(X, y)

    if st.button("🚀 XÁC ĐỊNH GIÁ TRỊ NGAY"):
        in_dict = {col: 0 for col in f_cols}
        in_dict["year"], in_dict["odo_numeric"] = u_year, u_odo
        in_dict["condition "] = u_cond # Đảm bảo khớp tên cột trong file
        in_dict["is_repaired"] = 1 if u_rep == "Đã sửa chữa/Thay thế" else 0
        
        for k in [f"brand_{u_brand}", f"model_{u_model}", f"location_{u_loc}"]:
            if k in in_dict: in_dict[k] = 1
        
        pred = model.predict(pd.DataFrame([in_dict]))[0]
        st.markdown(f"""
            <div class="result-box">
                <h2 style="color:#1E88E5;">GIÁ DỰ ĐOÁN</h2>
                <div class="price-text">{max(pred, 0):,.0f} VNĐ</div>
            </div>
        """, unsafe_allow_html=True)
        st.balloons()
