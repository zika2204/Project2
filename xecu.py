import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split # Thêm thư viện này theo ảnh bạn gửi
from sklearn import metrics # Để đo lường sai sót

# =========================================
# CẤU HÌNH APP & CSS (Giữ nguyên của bạn)
# =========================================
st.set_page_config(page_title="MOTO CŨ VN", page_icon="🏍️", layout="centered")
st.markdown("""<style>.main {padding-top: 20px;} .title {text-align: center; font-size: 52px; font-weight: bold; color: #E53935;} .subtitle {text-align: center; color: gray; font-size: 18px; margin-bottom: 35px;} .result-box {background-color: #f5f5f5; padding: 30px; border-radius: 18px; margin-top: 25px; text-align: center;}</style>""", unsafe_allow_html=True)

# =========================================
# LOAD DATA (Tối ưu lại phần xử lý)
# =========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("xecu.csv")
        df.columns = df.columns.str.strip().str.lower()

        # Xử lý Price & Odo
        df["price_numeric"] = df["price"].astype(str).str.replace('[",]', '', regex=True).astype(float)
        df["odo_numeric"] = df["odo"].astype(str).str.replace('[.,]', '', regex=True).astype(float)
        
        # Xử lý Condition & Repaired
        df["condition"] = pd.to_numeric(df["condition"], errors="coerce")
        df["repaired_parts"] = df["repaired_parts"].astype(str).str.lower().map({"yes": 1, "no": 0, "có": 1, "không": 0})
        
        # One Hot Encoding cho Location
        df = pd.get_dummies(df, columns=["location"])
        return df.dropna()
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu: {e}")
        return None

df = load_data()

# =========================================
# HEADER
# =========================================
st.markdown('<p class="title">🏍️ MOTO CŨ VN</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI dự đoán giá xe máy cũ (Chuẩn Train/Test Split)</p>', unsafe_allow_html=True)

if df is not None:
    # --- PHẦN NHẬP LIỆU (Giữ nguyên logic của bạn) ---
    location_columns = [col for col in df.columns if col.startswith("location_")]
    all_brands = sorted(df["brand"].unique())
    selected_brand = st.selectbox("Hãng xe", all_brands)
    
    all_models = sorted(df[df["brand"] == selected_brand]["model"].unique())
    selected_model = st.selectbox("Dòng xe", all_models)

    col1, col2 = st.columns(2)
    with col1:
        input_year = st.number_input("Năm sản xuất", min_value=2010, max_value=2026, value=2022)
    with col2:
        input_odo = st.number_input("Số KM đã chạy", min_value=0, value=5000, step=500)

    input_condition = st.slider("Tình trạng xe (0 - 10)", 0, 10, 8)
    repaired_input = st.radio("Xe đã thay phụ tùng?", ["Không", "Có"])
    repaired_value = 1 if repaired_input == "Có" else 0

    all_locations = [col.replace("location_", "") for col in location_columns]
    selected_location = st.selectbox("Khu vực", all_locations)

    # =========================================
    # THỰC HIỆN ML THEO CÁCH BẠN HỌC (ẢNH GỬI)
    # =========================================
    # Gom dữ liệu để máy học (Cùng model hoặc cùng hãng để dữ liệu phong phú hơn)
    data_train_all = df[(df["model"] == selected_model) | (df["brand"] == selected_brand)].drop_duplicates()

    if len(data_train_all) >= 10: # Cần nhiều dữ liệu hơn một chút để chia tập Test
        feature_columns = ["year", "odo_numeric", "condition", "repaired_parts"] + location_columns
        X = data_train_all[feature_columns]
        y = data_train_all["price_numeric"]

        # --- BƯỚC CHIA DỮ LIỆU THEO ẢNH ---
        # 80% để học (Train), 20% để kiểm tra (Test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Khởi tạo mô hình
        model_ai = LinearRegression()

        # Máy bắt đầu học trên tập Train
        model_ai.fit(X_train, y_train)

        # Kiểm tra sai sót trên tập Test
        y_pred_test = model_ai.predict(X_test)
        mae = metrics.mean_absolute_error(y_test, y_pred_test) # Sai số trung bình (VNĐ)
        r2 = metrics.r2_score(y_test, y_pred_test) # Độ chính xác (%)

        # --- NÚT DỰ ĐOÁN ---
        if st.button("💰 Dự đoán giá"):
            # Chuẩn bị dữ liệu input
            input_dict = {"year": input_year, "odo_numeric": input_odo, "condition": input_condition, "repaired_parts": repaired_value}
            for col in location_columns: input_dict[col] = 0
            selected_col = f"location_{selected_location}"
            if selected_col in input_dict: input_dict[selected_col] = 1

            X_new = pd.DataFrame([input_dict])
            prediction = model_ai.predict(X_new)[0]
            final_price = max(prediction, 0)

            # HIỂN THỊ KẾT QUẢ
            st.markdown(f"""<div class="result-box"><h2>💵 Giá dự đoán</h2><h1 style="color:#E53935;">{final_price:,.0f} VNĐ</h1>
                        <p style="color:gray;">Độ tin cậy của mô hình: <b>{r2:.2%}</b><br>
                        Sai số dự kiến: +/- {mae:,.0f} VNĐ</p></div>""", unsafe_allow_html=True)
            
            if r2 < 0.5:
                st.warning("⚠️ Dữ liệu về dòng xe này còn ít hoặc biến động lớn, kết quả chỉ mang tính chất tham khảo.")
    else:
        st.warning("❌ Dữ liệu quá ít để thực hiện chia tập Train/Test theo chuẩn ML. Hãy thêm dữ liệu vào file CSV.")

    with st.expander("📋 Xem dữ liệu tham khảo"):
        st.dataframe(df.head(10))
