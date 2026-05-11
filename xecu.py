import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# =====================================
# CẤU HÌNH GIAO DIỆN
# =====================================
st.set_page_config(
    page_title="MOTO CŨ VN",
    page_icon="🏍️",
    layout="centered"
)

# =====================================
# CSS
# =====================================
st.markdown("""
    <style>
    .main {
        padding-top: 20px;
    }

    .title {
        text-align: center;
        font-size: 45px;
        font-weight: bold;
        color: #E53935;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: gray;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 20px;
        border-radius: 15px;
        background-color: #f3f3f3;
        margin-top: 20px;
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

        # Xử lý giá
        df["price_numeric"] = (
            df["price"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace('"', "", regex=False)
            .astype(float)
        )

        # Xử lý ODO
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


df = load_data()

# =====================================
# HEADER
# =====================================
st.markdown(
    '<p class="title">🏍️ MOTO CŨ VN</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">AI dự đoán giá xe máy cũ tại Việt Nam</p>',
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
    # INPUT NĂM & ODO
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

        X_train = data_train[
            ["year", "odo_numeric"]
        ]

        y_train = data_train[
            "price_numeric"
        ]

        model_ai = LinearRegression()

        model_ai.fit(X_train, y_train)

        # =====================================
        # BUTTON
        # =====================================
        if st.button("💰 Dự đoán giá"):

            # Dữ liệu mới
            X_new = pd.DataFrame(
                [[input_year, input_odo]],
                columns=[
                    "year",
                    "odo_numeric"
                ]
            )

            # Predict
            prediction = model_ai.predict(X_new)[0]

            final_price = max(prediction, 0)

            # =====================================
            # TÍNH GIẢM GIÁ
            # =====================================
            newest_year = data_train["year"].max()

            years_used = newest_year - input_year

            depreciation_year = years_used * 0.05 * final_price

            depreciation_odo = (
                input_odo / 1000
            ) * 120000

            total_depreciation = (
                depreciation_year
                + depreciation_odo
            )

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

                <hr>

                <h3>📉 Mức giảm giá</h3>

                <p>
                • Theo số năm sử dụng:
                <b>{depreciation_year:,.0f} VNĐ</b>
                </p>

                <p>
                • Theo số KM đã chạy:
                <b>{depreciation_odo:,.0f} VNĐ</b>
                </p>

                <p>
                • Tổng mức khấu hao:
                <b>{total_depreciation:,.0f} VNĐ</b>
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

    else:
        st.warning(
            "Không đủ dữ liệu để AI học."
        )

    # =====================================
    # XEM DATA
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
