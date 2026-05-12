import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# =========================================
# CẤU HÌNH GIAO DIỆN XANH DƯƠNG TOÀN DIỆN
# =========================================
st.set_page_config(page_title="MOTO CŨ VN - BLUE", page_icon="🏍️", layout="centered")

st.markdown("""
<style>
    /* 1. Màu nền và Font */
    .stApp { background-color: #f8fbff; }

    /* 2. Tiêu đề MOTO CŨ VN cực đại (Đã xóa logo) */
    .title { 
        text-align: center;
        font-size: 100px !important; 
        font-weight: 900 !important; 
        color: #1565C0; 
        margin-top: 20px;
        margin-bottom: 0px;
        line-height: 1;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.1);
    }
    
    .subtitle { 
        text-align: center; 
        color: #546E7A; 
        font-size: 22px; 
        margin-bottom: 40px; 
        font-style: italic;
    }

    /* 3. Đổi màu thanh kéo Slider sang Xanh Dương */
    .stSlider [data-baseweb="slider"] [aria-valuemax] {
        background-color: #1E88E5;
    }
    .stSlider [data-baseweb="thumb"] {
        background-color: #1565C0;
        border: 2px solid #ffffff;
    }

    /* 4. Đổi màu Radio Button sang Xanh Dương */
    div[data-testid="stRadio"] label div[data-testid="stMarkdownContainer"] p {
        color: #0D47A1;
        font-weight: bold;
    }
    /* Màu khi được chọn */
    div[data-testid="stRadio"] div[role="radiogroup"] > div[data-bv-tabindex="0"] {
        background-color: #E3F2FD;
        border-radius: 10px;
    }

    /* 5. Khung hiển thị giá dự đoán */
    .result-box { 
        background-color: #E3F2FD; 
        padding: 40px; 
        border-radius: 25px; 
        border: 4px solid #1E88E5; 
        text-align: center; 
        margin-top: 20px;
    }
    
    .price-text { color: #0D47A1; font-size: 55px; font-weight: bold; }

    /* 6. Nút bấm định giá chuyên nghiệp */
    div.stButton > button {
        background-color: #1E88E5 !important;
        color: white !important;
        font-size: 24px !important;
        font-weight: bold !important;
        border-radius: 15px !important;
        height: 65px !important;
        width: 100% !important;
        border: none !important;
        transition: 0.3s;
        box-shadow: 0px 4px 10px rgba(30, 136, 229, 0.3);
    }
    div.stButton > button:hover {
        background-color: #0D47A1 !important;
        transform: scale(1.01);
    }
</style>
""", unsafe_allow_html=True)

# =========================================
# HÀM XỬ LÝ DỮ LIỆU
# =========================================
@st.cache_data
def load_and_clean_data():
    try:
        df = pd.read_csv("xecu.csv")
        df.columns = df.columns.str.strip().str.lower()
        
        def clean_numeric(value):
            if pd.isna(value): return 0.0
            clean_val = "".join(filter(str.isdigit, str(value)))
            return float(clean_val) if clean_val else 0.0

        df["price_numeric"] = df["price"].apply(clean_numeric)
        df["odo_numeric"] = df["odo"].apply(clean_numeric)
        df = df[df["price_numeric"] > 500000] 
        df["is_repaired"] = df["repaired_parts"].astype(str).str.lower().str.contains("yes|có").astype(int)
        df["condition"] = pd.to_numeric(df["condition"], errors="coerce").fillna(7)
        
        brands = sorted(df["brand"].unique())
        models = sorted(df["model"].unique())
        locs = sorted(df["location"].unique())
        df_ml = pd.get_dummies(df, columns=["brand", "model", "location"])
        return df_ml, brands, models, locs
    except Exception as e:
        st.error(f"Lỗi: {e}")
        return None, None, None, None

df_ml, brands, models, locations = load_and_clean_data()

# =========================================
# PHẦN TIÊU ĐỀ (KHÔNG LOGO)
# =========================================
st.markdown('<p class="title">MOTO CŨ VN</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Định giá xe máy cũ bằng Trí tuệ nhân tạo</p>', unsafe_allow_html=True)

# =========================================
# FORM NHẬP LIỆU
# =========================================
if df_ml is not None:
    with st.container():
        c1, c2 = st.columns(2)
        with c1:
            u_brand = st.selectbox("🎯 Hãng xe", brands)
            u_year = st.number_input("📅 Năm sản xuất", 2010, 2026, 2023)
            u_cond = st.slider("✨ Độ mới (1-10)", 1, 10, 8) # Màu xanh đã được CSS chỉnh
        with c2:
            u_model = st.selectbox("🏍️ Dòng xe", models)
            u_odo = st.number_input("🛣️ Số KM đã đi", 0, 300000, 5000, step=500)
            u_rep = st.radio("🛠️ Phụ tùng", ["Còn Zin (Chưa sửa)", "Đã thay/Sửa chữa"]) # Màu xanh đã được CSS chỉnh
            u_loc = st.selectbox("📍 Khu vực", locations)

    # =========================================
    # ML LOGIC
    # =========================================
    feature_cols = [c for c in df_ml.columns if any(x in c for x in ["brand_", "model_", "location_"])]
    feature_cols += ["year", "odo_numeric", "condition", "is_repaired"]
    
    X = df_ml[feature_cols]
    y = df_ml["price_numeric"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_ai = LinearRegression().fit(X_train, y_train)

    st.write("") 
    if st.button("🚀 XÁC ĐỊNH GIÁ TRỊ NGAY"):
        in_dict = {col: 0 for col in feature_cols}
        in_dict["year"], in_dict["odo_numeric"] = u_year, u_odo
        in_dict["condition"] = u_cond
        in_dict["is_repaired"] = 1 if u_rep == "Đã thay/Sửa chữa" else 0
        
        if f"brand_{u_
