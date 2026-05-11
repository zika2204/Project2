import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# =====================================
# CẤU HÌNH TRANG
# =====================================
st.set_page_config(
    page_title="MOTO CŨ VN",
    page_icon="🏍️",
    layout="centered"
)

# =====================================
# CSS GIAO DIỆN
# =====================================
st.markdown("""
<style>

.main {
    padding-top: 20px;
}

.title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: #E53935;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: gray;
    margin-bottom: 35px;
}

.result-box {
    background-color: #f5f5f5;
    padding: 25px;
    border-radius: 15px;
    margin-top: 25px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# LOAD DATA
# =====================================
@st.cache_data
def load_data():

    try:

        # Đọc file CSV
        df = pd.read_csv("xecu.csv")

        # Chuẩn hóa tên cột
        df.columns = df.columns.str.strip().str.lower()

        # =====================================
        # XỬ LÝ GIÁ
        # =====================================
        df["price_numeric"] = (
            df["price"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace('"', "", regex=False)
            .astype(float)
        )

        # =====================================
        # XỬ LÝ ODO
        # =====================================
        df["odo_numeric"] = (
            df["odo"]
            .astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", "", regex=False)
            .astype(float)
        )

        # Xóa dữ liệu lỗi
        df = df.dropna()

        return df

    except Exception as e:
        st.error(f"Lỗi tải dữ liệu: {e}")
        return None


# =====================================
# LOAD DATA
# =====================================
df = load_data()

# =====================================
# HEADER
# =====================================
st.markdown(
    '<p class="title">🏍️ MOTO CŨ VN</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">AI dự đoán giá xe máy cũ bằng Machine Learning</p>',
    unsafe_allow_html=True
)

# =====================================
# MAIN APP
# =====================================
if df is not None:

    st.subheader("📌 Nhập thông tin xe")

    # =====================================
    # CHỌN HÃNG XE
    # =====================================
    all_brands = sorted(df["brand"].unique())

    selected_brand = st.selectbox(
        "Hãng xe",
        all_brands
    )

    # =====================================
    # CHỌN DÒNG XE
    # =====================================
    all_models = sorted(
        df[df["brand"] == selected_brand]["model"].unique()
    )

    selected_model = st.selectbox(
        "Dòng xe",
        all_models
    )

    # =====================================
    # NHẬP NĂM & ODO
    # =====================================
    col1, col2 = st.columns(2)

    with col1:
        input_year = st.number_input(
            "Năm sản xuất",
            min_value=2010,
            max_value=2026,
            value=2022
        )

    with col2:
        input_odo = st.number_input(
            "Số KM đã chạy",
            min_value=0,
            value=5000,
            step=500
        )

    # =====================================
    # DATA TRAIN
    # =====================================
    data_same_model = df[
        df["model"] == selected_model
    ]

    data_same_brand = df[
        df["brand"] == selected_brand
    ]

    data_train = pd.concat([
        data_same_model,
        data_same_brand
    ])

    data_train = data_train.drop_duplicates()

    # =====================================
    # TRAIN MODEL
    # =====================================
    if len(data_train) >= 5:

        # Feature
        X_train = data_train[
            [
                "year",
                "odo_numeric"
            ]
        ]

        # Target
        y_train = data_train[
            "price_numeric"
        ]

        # Model AI
        model_ai = LinearRegression()

        # Huấn luyện
        model_ai.fit(X_train, y_train)

        # =====================================
        # BUTTON DỰ ĐOÁN
        # =====================================
        if st.button("💰 Dự đoán giá"):

            # Input mới
            X_new = pd.DataFrame(
                [[
                    input_year,
                    input_odo
                ]],
                columns=[
                    "year",
                    "odo_numeric"
                ]
            )

            # Predict
            prediction = model_ai.predict(X_new)[0]

            # Không cho giá âm
            final_price = max(prediction, 0)

            # =====================================
            # HIỂN THỊ KẾT QUẢ
            # =====================================
            st.markdown(
                f"""
                <div class="result-box">

                <h2>💵 Giá dự đoán</h2>

                <h1 style="color:#E53935;">
                {final_price:,.0f} VNĐ
                </h1>

                <p style="color:gray;">
                Giá được AI dự đoán dựa trên
                năm sản xuất và số KM đã chạy.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

    else:
        st.warning(
            "❌ Không đủ dữ liệu để AI học."
        )

    # =====================================
    # XEM DỮ LIỆU
    # =====================================
    with st.expander("📋 Xem dữ liệu tham khảo"):

        st.dataframe(
            df[
                [
                    "brand",
                    "model",
                    "year",
                    "odo",
                    "price"
                ]
            ]
        )

else:
    st.info(
        "Hãy kiểm tra file xecu.csv."
    )
