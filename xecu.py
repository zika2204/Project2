import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# =========================================
# CẤU HÌNH APP & CSS (TĂNG SIZE CHỮ TIÊU ĐỀ)
# =========================================
st.set_page_config(page_title="MOTO CŨ VN", page_icon="🏍️", layout="centered")

st.markdown("""
<style>
    .main { padding-top: 10px; }
    /* Tăng size chữ tiêu đề lên 70px và đổ bóng cho chuyên nghiệp */
    .title {
        text-align: center;
        font-size: 75px; 
        font-weight: 900;
        color: #E53935;
        margin-bottom: 0px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 20px;
        margin-bottom: 40px;
    }
    .result-box {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 20px;
        border: 3px solid #E53935;
        text-align: center;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.05);
    }
    .price-text {
        color: #E53935;
        font-size: 45px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# =========================================
# LOAD & XỬ LÝ DỮ LIỆU
# =========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("xecu.csv")
        df.columns = df.columns.str.strip().str.lower()
        # Dọn dẹp dữ liệu số
        df["price_numeric"] = df["price"].astype(str).str.replace('[",]', '', regex=True).astype(float)
        df["odo_numeric"] = df["odo"].astype(str).str.replace('[.,]', '', regex=True).astype(float)
        df["condition"] = pd.to_numeric(df["condition"], errors="coerce")
        # Logic phụ tùng: Zin = 0, Đã thay = 1
        df["is_repaired"] = df["repaired_parts"].astype(str).str.lower().str.strip().map({
            "yes": 1, "có": 1, "no": 0, "không": 0
        }).fillna(0)
        # Location Encoding
        df = pd.get_dummies(df, columns=["location"])
        return df.dropna(subset=['price_numeric', 'year'])
    except Exception as e:
        st.error(f"Lỗi: {e}")
        return None

df = load_data()

# =========================================
# GIAO DIỆN CHÍNH
# =========================================
st.markdown('<p class="title">MOTO CŨ VN</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Hệ thống định giá xe máy cũ thông minh</p>', unsafe_allow_html=True)

if df is not None:
    # Chọn xe
    brand_list = sorted(df["brand"].unique())
    col_a, col_b = st.columns(2)
    with col_a:
        selected_brand = st.selectbox("Hãng xe", brand_list)
    with col_b:
        model_list = sorted(df[df["brand"] == selected_brand]["model"].unique())
        selected_model = st.selectbox("Dòng xe", model_list)

    # Nhập thông số
    col1, col2 = st.columns(2)
    with col1:
        input_year = st.number_input("Năm sản xuất", 2010, 2026, 2022)
        input_condition = st.slider("Độ mới xe (1-10)", 1, 10, 8)
    with col2:
        input_odo = st.number_input("Số KM đã chạy", 0, 200000, 5000)
        repaired_input = st.radio("Tình trạng phụ tùng:", ["Còn Zin (Chưa thay)", "Đã thay/Sửa chữa"])

    repaired_val = 1 if repaired_input == "Đã thay/Sửa chữa" else 0
    
    loc_cols = [c for c in df.columns if c.startswith("location_")]
    selected_loc = st.selectbox("Khu vực", [c.replace("location_", "") for c in loc_cols])

    # =========================================
    # HUẤN LUYỆN ML (ẨN ĐỘ CHÍNH XÁC)
    # =========================================
    data_train = df[(df["model"] == selected_model) | (df["brand"] == selected_brand)]

    if len(data_train) >= 5:
        features = ["year", "odo_numeric", "condition", "is_repaired"] + loc_cols
        X = data_train[features]
        y = data_train["price_numeric"]

        # Chia dữ liệu chuẩn bài học
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        if st.button("🚀 BẮT ĐẦU ĐỊNH GIÁ", use_container_width=True):
            # Input data chuẩn bị dự đoán
            in_data = {"year": input_year, "odo_numeric": input_odo, "condition": input_condition, "is_repaired": repaired_val}
            for c in loc_cols: in_data[c] = 1 if c == f"location_{selected_loc}" else 0
            
            X_new = pd.DataFrame([in_data])
            pred = model.predict(X_new)[0]
            
            # Hiển thị kết quả (Đã bỏ phần Độ chính xác)
            st.markdown(f"""
            <div class="result-box">
                <p style="font-size: 20px; color: #666; margin-bottom: 5px;">Giá trị dự kiến từ AI</p>
                <div class="price-text">{max(pred, 0):,.0f} VNĐ</div>
                <p style="font-size: 14px; color: #999; margin-top: 10px;">
                    * Mức giá có thể thay đổi tùy theo thỏa thuận thực tế và biển số xe.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info(f"💡 **AI tư vấn:** Xe {selected_model} đời {input_year} hiện đang có nhu cầu cao trên thị trường.")
    else:
        st.warning("Dữ liệu không đủ để AI thực hiện tính toán.")

    with st.expander("📋 Xem danh sách xe tham khảo"):
        st.dataframe(df[['brand', 'model', 'year', 'odo', 'price']].head(15))
