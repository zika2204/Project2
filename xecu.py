import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# =========================================
# CẤU HÌNH & GIAO DIỆN
# =========================================
st.set_page_config(page_title="MOTO CŨ VN - AI Pricing", page_icon="🏍️", layout="centered")

st.markdown("""
<style>
    .title {text-align: center; font-size: 45px; font-weight: bold; color: #E53935;}
    .result-box {background-color: #ffffff; padding: 25px; border-radius: 15px; border: 2px solid #E53935; text-align: center;}
</style>
""", unsafe_allow_html=True)

# =========================================
# LOAD & XỬ LÝ DỮ LIỆU
# =========================================
@st.cache_data
def load_data():
    try:
        # Đọc file (đảm bảo file xecu.csv có trên github)
        df = pd.read_csv("xecu.csv")
        df.columns = df.columns.str.strip().str.lower()

        # 1. Xử lý Giá & Odo (Bỏ ký tự lạ)
        df["price_numeric"] = df["price"].astype(str).str.replace('[",]', '', regex=True).astype(float)
        df["odo_numeric"] = df["odo"].astype(str).str.replace('[.,]', '', regex=True).astype(float)
        
        # 2. Xử lý Condition
        df["condition"] = pd.to_numeric(df["condition"], errors="coerce")
        
        # 3. Xử lý Phụ tùng (Repaired) - ĐƯA VỀ LOGIC CỦA BẠN:
        # Không thay (Zin) = 0, Có thay = 1
        df["repaired_parts"] = df["repaired_parts"].astype(str).str.lower().str.strip()
        df["is_repaired"] = df["repaired_parts"].map({
            "yes": 1, "có": 1, 
            "no": 0, "không": 0
        }).fillna(0)

        # 4. One-Hot Encoding cho Location
        df = pd.get_dummies(df, columns=["location"])
        
        return df.dropna(subset=['price_numeric', 'odo_numeric', 'year', 'condition'])
    except Exception as e:
        st.error(f"Lỗi dữ liệu: {e}")
        return None

df = load_data()

# =========================================
# GIAO DIỆN NHẬP LIỆU
# =========================================
st.markdown('<p class="title">🏍️ MOTO CŨ VN</p>', unsafe_allow_html=True)

if df is not None:
    brand_list = sorted(df["brand"].unique())
    col_a, col_b = st.columns(2)
    with col_a:
        selected_brand = st.selectbox("Hãng xe", brand_list)
    with col_b:
        model_list = sorted(df[df["brand"] == selected_brand]["model"].unique())
        selected_model = st.selectbox("Dòng xe", model_list)

    col1, col2 = st.columns(2)
    with col1:
        input_year = st.number_input("Năm sản xuất (Càng cao giá càng tăng)", 2010, 2026, 2022)
        input_condition = st.slider("Độ mới xe (0: Nát - 10: Như mới)", 0, 10, 8)
    with col2:
        input_odo = st.number_input("Số KM đã chạy (Càng cao giá càng giảm)", 0, 200000, 5000)
        repaired_input = st.radio("Tình trạng phụ tùng:", ["Còn Zin (Chưa thay)", "Đã thay/Sửa chữa"])

    repaired_val = 1 if repaired_input == "Đã thay/Sửa chữa" else 0
    
    loc_cols = [c for c in df.columns if c.startswith("location_")]
    selected_loc = st.selectbox("Khu vực", [c.replace("location_", "") for c in loc_cols])

    # =========================================
    # MACHINE LEARNING (TRAIN/TEST SPLIT)
    # =========================================
    # Gom dữ liệu xe cùng loại để máy học quy luật
    data_train = df[(df["model"] == selected_model) | (df["brand"] == selected_brand)]

    if len(data_train) >= 8:
        features = ["year", "odo_numeric", "condition", "is_repaired"] + loc_cols
        X = data_train[features]
        y = data_train["price_numeric"]

        # Chia 80% học, 20% thi (Theo ảnh bài học L3)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        if st.button("🚀 ĐỊNH GIÁ XE NGAY"):
            # Dự đoán
            in_data = {"year": input_year, "odo_numeric": input_odo, "condition": input_condition, "is_repaired": repaired_val}
            for c in loc_cols: in_data[c] = 1 if c == f"location_{selected_loc}" else 0
            
            X_new = pd.DataFrame([in_data])
            pred = model.predict(X_new)[0]
            
            # Logic kiểm tra: Đảm bảo giá xe zin > giá xe đã thay phụ tùng nếu các biến khác bằng nhau
            # (Linear Regression sẽ tự học điều này nếu dữ liệu chuẩn)
            
            st.markdown(f"""
            <div class="result-box">
                <h3 style="color:gray;">Giá trị ước tính</h3>
                <h1 style="color:#E53935;">{max(pred, 0):,.0f} VNĐ</h1>
                <p>Độ chính xác mô hình: {model.score(X_test, y_test):.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Giải thích logic cho người dùng
            st.write("---")
            st.info(f"**Giải thích từ AI:** Xe đời {input_year} với odo {input_odo:,}km được định giá dựa trên xu hướng giảm giá theo thời gian và độ hao mòn linh kiện.")
    else:
        st.warning("Dữ liệu dòng xe này quá ít, AI chưa thể học được quy luật.")
