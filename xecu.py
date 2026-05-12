import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# =========================================
# CẤU HÌNH GIAO DIỆN
# =========================================
st.set_page_config(page_title="MOTO CŨ VN - BLUE", page_icon="🏍️", layout="centered")

# CSS để chỉnh màu xanh dương và phóng to tiêu đề
st.markdown("""
<style>
    /* Màu nền nhẹ nhàng cho toàn trang */
    .stApp { background-color: #f8fbff; }

    /* Container cho Logo và Tiêu đề */
    .header-container {
        text-align: center;
        padding-bottom: 20px;
    }

    /* Logo hình tròn hoặc bo góc */
    .logo-img {
        width: 150px;
        margin-bottom: -10px;
    }

    /* Tiêu đề MOTO CŨ VN cực đại */
    .title { 
        font-size: 95px !important; 
        font-weight: 900 !important; 
        color: #1565C0; 
        margin-top: 0px;
        line-height: 1;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.1);
    }
    
    .subtitle { 
        text-align: center; 
        color: #546E7A; 
        font-size: 22px; 
        margin-bottom: 30px; 
        font-style: italic;
    }

    /* Khung hiển thị giá dự đoán */
    .result-box { 
        background-color: #E3F2FD; 
        padding: 40px; 
        border-radius: 25px; 
        border: 4px solid #1E88E5; 
        text-align: center; 
        margin-top: 20px;
    }
    
    .price-text { color: #0D47A1; font-size: 55px; font-weight: bold; }

    /* Nút bấm định giá màu xanh */
    div.stButton > button {
        background-color: #1E88E5 !important;
        color: white !important;
        font-size: 24px !important;
        font-weight: bold !important;
        border-radius: 15px !important;
        height: 60px !important;
        width: 100% !important;
        border: none !important;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #0D47A1 !important;
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# =========================================
# HÀM XỬ LÝ DỮ LIỆU (FIX LỖI FLOAT)
# =========================================
@st.cache_data
def load_and_clean_data():
    try:
        df = pd.read_csv("xecu.csv")
        df.columns = df.columns.str.strip().str.lower()
        
        def clean_numeric(value):
            if pd.isna(value): return 0.0
            # Lọc lấy duy nhất các chữ số (bỏ qua "đ", ".", ",")
            clean_val = "".join(filter(str.isdigit, str(value)))
            return float(clean_val) if clean_val else 0.0

        df["price_numeric"] = df["price"].apply(clean_numeric)
        df["odo_numeric"] = df["odo"].apply(clean_numeric)
        
        # Chỉ lấy dữ liệu xe có giá thực tế
        df = df[df["price_numeric"] > 500000] 

        # Chuẩn hóa zin/sửa
        df["is_repaired"] = df["repaired_parts"].astype(str).str.lower().str.contains("yes|có").astype(int)
        df["condition"] = pd.to_numeric(df["condition"], errors="coerce").fillna(7)
        
        brands = sorted(df["brand"].unique())
        models = sorted(df["model"].unique())
        locs = sorted(df["location"].unique())
        
        # One-Hot Encoding
        df_ml = pd.get_dummies(df, columns=["brand", "model", "location"])
        
        return df_ml, brands, models, locs
    except Exception as e:
        st.error(f"Lỗi: {e}")
        return None, None, None, None

df_ml, brands, models, locations = load_and_clean_data()

# =========================================
# PHẦN HIỂN THỊ LOGO VÀ TIÊU ĐỀ
# =========================================
# Nếu bạn có file logo.png, hãy thay link bên dưới thành: src="data:image/png;base64,..." hoặc link online
st.markdown("""
    <div class="header-container">
        <img src="https://cdn-icons-png.flaticon.com/512/3198/3198338.png" class="logo-img">
        <p class="title">MOTO CŨ VN</p>
        <p class="subtitle">Trí tuệ nhân tạo định giá xe máy cũ Việt Nam</p>
    </div>
""", unsafe_allow_html=True)

# =========================================
# FORM NHẬP LIỆU
# =========================================
if df_ml is not None:
    with st.container():
        c1, c2 = st.columns(2)
        with c1:
            u_brand = st.selectbox("🎯 Chọn Hãng xe", brands)
            u_year = st.number_input("📅 Năm sản xuất", 2010, 2026, 2023)
            u_cond = st.slider("✨ Độ mới của xe (1-10)", 1, 10, 8)
        with c2:
            u_model = st.selectbox("🏍️ Chọn Dòng xe", models)
            u_odo = st.number_input("🛣️ Số KM đã đi (Odo)", 0, 300000, 5000, step=500)
            u_rep = st.radio("🛠️ Tình trạng máy móc", ["Còn Zin (Chưa sửa)", "Đã thay/Sửa chữa"])
            u_loc = st.selectbox("📍 Khu vực mua bán", locations)

    # =========================================
    # LOGIC MÁY HỌC (LINEAR REGRESSION)
    # =========================================
    feature_cols = [c for c in df_ml.columns if any(x in c for x in ["brand_", "model_", "location_"])]
    feature_cols += ["year", "odo_numeric", "condition", "is_repaired"]
    
    X = df_ml[feature_cols]
    y = df_ml["price_numeric"]

    # Train/Test split theo ảnh bài L3 bạn học
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_ai = LinearRegression()
    model_ai.fit(X_train, y_train)

    st.write("") # Khoảng cách
    if st.button("🚀 XÁC ĐỊNH GIÁ TRỊ NGAY"):
        # Chuẩn bị dữ liệu input
        in_dict = {col: 0 for col in feature_cols}
        in_dict["year"] = u_year
        in_dict["odo_numeric"] = u_odo
        in_dict["condition"] = u_cond
        in_dict["is_repaired"] = 1 if u_rep == "Đã thay/Sửa chữa" else 0
        
        if f"brand_{u_brand}" in in_dict: in_dict[f"brand_{u_brand}"] = 1
        if f"model_{u_model}" in in_dict: in_dict[f"model_{u_model}"] = 1
        if f"location_{u_loc}" in in_dict: in_dict[f"location_{u_loc}"] = 1
        
        # Dự đoán
        prediction = model_ai.predict(pd.DataFrame([in_dict]))[0]

        # Hiển thị kết quả Blue Theme
        st.markdown(f"""
        <div class="result-box">
            <p style="font-size: 22px; color: #1565C0; font-weight: bold; margin-bottom: 10px;">GIÁ TRỊ ƯỚC TÍNH TỪ AI</p>
            <div class="price-text">{max(prediction, 0):,.0f} VNĐ</div>
            <p style="color: #455A64; font-size: 15px; margin-top: 15px;">
                Dòng xe: <b>{u_model}</b> | Đời: <b>{u_year}</b> | Khu vực: <b>{u_loc}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.balloons() # Hiệu ứng chúc mừng khi ra kết quả
