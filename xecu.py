import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# =========================================
# CẤU HÌNH GIAO DIỆN
# =========================================
st.set_page_config(page_title="MOTO CŨ VN", page_icon="🏍️", layout="centered")

st.markdown("""
<style>
    .title { text-align: center; font-size: 75px; font-weight: 900; color: #E53935; margin-bottom: 0px; }
    .subtitle { text-align: center; color: #555; font-size: 20px; margin-bottom: 40px; }
    .result-box { background-color: #ffffff; padding: 30px; border-radius: 20px; border: 3px solid #E53935; text-align: center; }
    .price-text { color: #E53935; font-size: 45px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# =========================================
# LOAD & XỬ LÝ DỮ LIỆU TỔNG THỂ
# =========================================
@st.cache_data
def load_full_data():
    try:
        df = pd.read_csv("xecu.csv")
        df.columns = df.columns.str.strip().str.lower()
        
        # Làm sạch số liệu
        df["price_numeric"] = df["price"].astype(str).str.replace('[",]', '', regex=True).astype(float)
        df["odo_numeric"] = df["odo"].astype(str).str.replace('[.,]', '', regex=True).astype(float)
        df["condition"] = pd.to_numeric(df["condition"], errors="coerce")
        df["is_repaired"] = df["repaired_parts"].astype(str).str.lower().str.strip().map({
            "yes": 1, "có": 1, "no": 0, "không": 0
        }).fillna(0)
        
        # Lưu lại danh sách hãng và model gốc để làm menu chọn
        original_brands = sorted(df["brand"].unique())
        original_models = sorted(df["model"].unique())
        original_locations = sorted(df["location"].unique())

        # CHUYỂN CHỮ THÀNH SỐ (One-Hot Encoding cho cả Brand, Model và Location)
        # Đây là bước quan trọng để máy học được "tên" xe
        df_ml = pd.get_dummies(df, columns=["brand", "model", "location"])
        
        return df_ml.dropna(subset=['price_numeric']), original_brands, original_models, original_locations
    except Exception as e:
        st.error(f"Lỗi: {e}")
        return None, None, None, None

df_ml, brands, models, locations = load_full_data()

# =========================================
# GIAO DIỆN
# =========================================
st.markdown('<p class="title">MOTO CŨ VN</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI Học máy toàn diện (Brand & Model)</p>', unsafe_allow_html=True)

if df_ml is not None:
    col_a, col_b = st.columns(2)
    with col_a:
        user_brand = st.selectbox("Chọn Hãng xe", brands)
    with col_b:
        # Chỉ hiện model thuộc hãng đã chọn cho dễ nhìn, nhưng máy vẫn học hết
        # Để đơn giản hóa giao diện, ta vẫn dùng menu nhưng máy sẽ dùng logic toàn file
        model_list = models # Có thể lọc theo hãng nếu muốn giao diện gọn hơn
        user_model = st.selectbox("Chọn Dòng xe", model_list)

    col1, col2 = st.columns(2)
    with col1:
        input_year = st.number_input("Năm sản xuất", 2010, 2026, 2022)
        input_condition = st.slider("Độ mới xe (1-10)", 1, 10, 8)
    with col2:
        input_odo = st.number_input("Số KM đã chạy", 0, 200000, 5000)
        repaired_input = st.radio("Tình trạng phụ tùng:", ["Còn Zin", "Đã sửa"])

    user_loc = st.selectbox("Khu vực", locations)

    # =========================================
    # HUẤN LUYỆN TOÀN BỘ DỮ LIỆU
    # =========================================
    # Lấy tất cả các cột dummy (brand_..., model_..., location_...)
    feature_cols = [c for c in df_ml.columns if any(x in c for x in ["brand_", "model_", "location_"])]
    feature_cols += ["year", "odo_numeric", "condition", "is_repaired"]
    
    X = df_ml[feature_cols]
    y = df_ml["price_numeric"]

    # Chia tập học và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_ai = LinearRegression()
    model_ai.fit(X_train, y_train)

    if st.button("🚀 ĐỊNH GIÁ TOÀN DIỆN", use_container_width=True):
        # Tạo dữ liệu đầu vào cho máy dự đoán
        input_dict = {col: 0 for col in feature_cols}
        input_dict["year"] = input_year
        input_dict["odo_numeric"] = input_odo
        input_dict["condition"] = input_condition
        input_dict["is_repaired"] = 1 if repaired_input == "Đã sửa" else 0
        
        # Bật các cột 0/1 tương ứng với lựa chọn của người dùng
        if f"brand_{user_brand}" in input_dict: input_dict[f"brand_{user_brand}"] = 1
        if f"model_{user_model}" in input_dict: input_dict[f"model_{user_model}"] = 1
        if f"location_{user_loc}" in input_dict: input_dict[f"location_{user_loc}"] = 1
        
        X_predict = pd.DataFrame([input_dict])
        prediction = model_ai.predict(X_predict)[0]

        st.markdown(f"""
        <div class="result-box">
            <p style="font-size: 20px; color: #666;">Giá trị dự đoán từ AI</p>
            <div class="price-text">{max(prediction, 0):,.0f} VNĐ</div>
            <p style="color: gray; font-size: 14px;">(Máy học dựa trên thương hiệu {user_brand} và dòng xe {user_model})</p>
        </div>
        """, unsafe_allow_html=True)
