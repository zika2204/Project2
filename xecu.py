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
# HÀM XỬ LÝ DỮ LIỆU (SỬA LỖI FLOAT)
# =========================================
@st.cache_data
def load_and_clean_data():
    try:
        # 1. Đọc file
        df = pd.read_csv("xecu.csv")
        df.columns = df.columns.str.strip().str.lower()
        
        # 2. HÀM DỌN RÁC SỐ (PRICE & ODO)
        # Regex [^\d] nghĩa là: "Xóa sạch tất cả những gì KHÔNG PHẢI là số"
        def clean_currency(value):
            if pd.isna(value): return 0.0
            clean_val = "".join(filter(str.isdigit, str(value)))
            return float(clean_val) if clean_val else 0.0

        df["price_numeric"] = df["price"].apply(clean_currency)
        df["odo_numeric"] = df["odo"].apply(clean_currency)
        
        # 3. Loại bỏ dữ liệu rác (Giá bằng 0 hoặc quá thấp)
        df = df[df["price_numeric"] > 500000] 

        # 4. Xử lý logic Condition & Repaired
        df["condition"] = pd.to_numeric(df["condition"], errors="coerce").fillna(5)
        df["is_repaired"] = df["repaired_parts"].astype(str).str.lower().str.contains("yes|có").astype(int)
        
        # 5. Lưu thông tin gốc để làm menu
        original_brands = sorted(df["brand"].unique())
        original_models = sorted(df["model"].unique())
        original_locations = sorted(df["location"].unique())

        # 6. One-Hot Encoding (Dùng cho ML để hiểu Brand/Model)
        df_ml = pd.get_dummies(df, columns=["brand", "model", "location"])
        
        return df_ml, original_brands, original_models, original_locations
    except Exception as e:
        st.error(f"Lỗi xử lý file: {e}")
        return None, None, None, None

df_ml, brands, models, locations = load_and_clean_data()

# =========================================
# GIAO DIỆN NGƯỜI DÙNG
# =========================================
st.markdown('<p class="title">MOTO CŨ VN</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI Học máy toàn diện dựa trên Hãng và Dòng xe</p>', unsafe_allow_html=True)

if df_ml is not None:
    col_a, col_b = st.columns(2)
    with col_a:
        user_brand = st.selectbox("Chọn Hãng xe", brands)
    with col_b:
        # Lọc model theo hãng để menu mượt hơn
        mask = [col for col in df_ml.columns if col.startswith(f"brand_{user_brand}")]
        # Lấy list model khả dụng của hãng đó
        available_models = sorted(models) # Để đơn giản ta hiện hết, AI sẽ tự lọc
        user_model = st.selectbox("Chọn Dòng xe", available_models)

    col1, col2 = st.columns(2)
    with col1:
        input_year = st.number_input("Năm sản xuất", 2010, 2026, 2022)
        input_condition = st.slider("Độ mới xe (1-10)", 1, 10, 8)
    with col2:
        input_odo = st.number_input("Số KM đã chạy", 0, 200000, 5000)
        repaired_input = st.radio("Tình trạng phụ tùng:", ["Còn Zin", "Đã sửa/Thay thế"])

    user_loc = st.selectbox("Khu vực", locations)

    # =========================================
    # HUẤN LUYỆN MÁY (LINEAR REGRESSION)
    # =========================================
    # Lấy danh sách các cột đã được dummies hóa
    feature_cols = [c for c in df_ml.columns if any(x in c for x in ["brand_", "model_", "location_"])]
    feature_cols += ["year", "odo_numeric", "condition", "is_repaired"]
    
    X = df_ml[feature_cols]
    y = df_ml["price_numeric"]

    # Chia Train/Test để máy không học vẹt
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_ai = LinearRegression()
    model_ai.fit(X_train, y_train)

    if st.button("🚀 XÁC ĐỊNH GIÁ TRỊ XE", use_container_width=True):
        # Chuẩn bị dữ liệu để đưa vào máy dự đoán
        input_dict = {col: 0 for col in feature_cols}
        input_dict["year"] = input_year
        input_dict["odo_numeric"] = input_odo
        input_dict["condition"] = input_condition
        input_dict["is_repaired"] = 1 if repaired_input == "Đã sửa/Thay thế" else 0
        
        # Kích hoạt các cột Brand/Model/Location tương ứng
        if f"brand_{user_brand}" in input_dict: input_dict[f"brand_{user_brand}"] = 1
        if f"model_{user_model}" in input_dict: input_dict[f"model_{user_model}"] = 1
        if f"location_{user_loc}" in input_dict: input_dict[f"location_{user_loc}"] = 1
        
        X_predict = pd.DataFrame([input_dict])
        prediction = model_ai.predict(X_predict)[0]

        # Hiển thị
        st.markdown(f"""
        <div class="result-box">
            <p style="font-size: 20px; color: #666;">Giá trị ước tính từ AI</p>
            <div class="price-text">{max(prediction, 0):,.0f} VNĐ</div>
            <p style="color: gray; font-size: 14px; margin-top:10px;">
                Dựa trên phân tích {len(df_ml)} mẫu xe thực tế trên thị trường.
            </p>
        </div>
        """, unsafe_allow_html=True)
