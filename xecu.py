import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# =========================================
# CẤU HÌNH APP
# =========================================
st.set_page_config(
    page_title="MOTO CŨ VN",
    page_icon="🏍️",
    layout="centered"
)

# =========================================
# CSS GIAO DIỆN
# =========================================
st.markdown("""
<style>

.main {
    padding-top: 20px;
}

.title {
    text-align: center;
    font-size: 52px;
    font-weight: bold;
    color: #E53935;
}

.subtitle {
    text-align: center;
    color: gray;
    font-size: 18px;
    margin-bottom: 35px;
}

.result-box {
    background-color: #f5f5f5;
    padding: 30px;
    border-radius: 18px;
    margin-top: 25px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# LOAD DATA
# =========================================
@st.cache_data
def load_data():

    try:

        # Đọc CSV
        df = pd.read_csv("xecu.csv")

        # Chuẩn hóa tên cột
        df.columns = df.columns.str.strip().str.lower()

        # =========================================
        # XỬ LÝ PRICE
        # =========================================
        df["price_numeric"] = (
            df["price"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace('"', "", regex=False)
            .astype(float)
        )

        # =========================================
        # XỬ LÝ ODO
        # =========================================
        df["odo_numeric"] = (
            df["odo"]
            .astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", "", regex=False)
            .astype(float)
        )

        # =========================================
        # XỬ LÝ CONDITION
        # =========================================
        df["condition"] = pd.to_numeric(
            df["condition"],
            errors="coerce"
        )

        # =========================================
        # XỬ LÝ REPAIRED PART
        # =========================================
        df["repaired_part"] = (
            df["repaired_part"]
            .astype(str)
            .str.lower()
        )

        df["repaired_part"] = (
            df["repaired_part"]
            .map({
                "yes": 1,
                "no": 0,
                "có": 1,
                "không": 0
            })
        )

        # =========================================
        # XỬ LÝ LOCATION
        # =========================================
        df["location"] = (
            df["location"]
            .astype(str)
        )

        # One Hot Encoding
        df = pd.get_dummies(
            df,
            columns=["location"]
        )

        # Xóa dòng lỗi
        df = df.dropna()

        return df

    except Exception as e:

        st.error(f"Lỗi tải dữ liệu: {e}")

        return None


# =========================================
# LOAD DATA
# =========================================
df = load_data()

# =========================================
# HEADER
# =========================================
st.markdown(
    '<p class="title">🏍️ MOTO CŨ VN</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">AI dự đoán giá xe máy cũ bằng Machine Learning</p>',
    unsafe_allow_html=True
)

# =========================================
# MAIN APP
# =========================================
if df is not None:

    st.subheader("📌 Nhập thông tin xe")

    # =========================================
    # LOCATION COLUMNS
    # =========================================
    location_columns = [
        col for col in df.columns
        if col.startswith("location_")
    ]

    # =========================================
    # BRAND
    # =========================================
    all_brands = sorted(
        df["brand"].unique()
    )

    selected_brand = st.selectbox(
        "Hãng xe",
        all_brands
    )

    # =========================================
    # MODEL
    # =========================================
    all_models = sorted(
        df[
            df["brand"] == selected_brand
        ]["model"].unique()
    )

    selected_model = st.selectbox(
        "Dòng xe",
        all_models
    )

    # =========================================
    # YEAR & ODO
    # =========================================
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

    # =========================================
    # CONDITION
    # =========================================
    input_condition = st.slider(
        "Tình trạng xe (0 - 10)",
        min_value=0,
        max_value=10,
        value=8
    )

    # =========================================
    # REPAIRED PART
    # =========================================
    repaired_input = st.radio(
        "Xe đã thay phụ tùng?",
        ["Không", "Có"]
    )

    repaired_value = (
        1 if repaired_input == "Có"
        else 0
    )

    # =========================================
    # LOCATION
    # =========================================
    all_locations = [
        col.replace("location_", "")
        for col in location_columns
    ]

    selected_location = st.selectbox(
        "Khu vực",
        all_locations
    )

    # =========================================
    # DATA TRAIN
    # =========================================
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

    # =========================================
    # TRAIN MODEL
    # =========================================
    if len(data_train) >= 5:

        # Feature columns
        feature_columns = [
            "year",
            "odo_numeric",
            "condition",
            "repaired_part"
        ] + location_columns

        # X train
        X_train = data_train[
            feature_columns
        ]

        # y train
        y_train = data_train[
            "price_numeric"
        ]

        # Model
        model_ai = LinearRegression()

        # Train
        model_ai.fit(
            X_train,
            y_train
        )

        # =========================================
        # BUTTON DỰ ĐOÁN
        # =========================================
        if st.button("💰 Dự đoán giá"):

            # Dictionary input
            input_data = {
                "year": input_year,
                "odo_numeric": input_odo,
                "condition": input_condition,
                "repaired_part": repaired_value
            }

            # Fill location columns
            for col in location_columns:

                input_data[col] = 0

            # Chọn location
            selected_col = (
                f"location_{selected_location}"
            )

            if selected_col in input_data:

                input_data[selected_col] = 1

            # DataFrame predict
            X_new = pd.DataFrame(
                [input_data]
            )

            # Predict
            prediction = (
                model_ai.predict(X_new)[0]
            )

            # Không cho giá âm
            final_price = max(
                prediction,
                0
            )

            # =========================================
            # HIỂN THỊ KẾT QUẢ
            # =========================================
            st.markdown(
                f"""
                <div class="result-box">

                <h2>💵 Giá dự đoán</h2>

                <h1 style="color:#E53935;">
                {final_price:,.0f} VNĐ
                </h1>

                <p style="color:gray;">
                Giá được AI dự đoán dựa trên:
                năm sản xuất, số KM đã chạy,
                tình trạng xe, phụ tùng và khu vực.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.warning(
            "❌ Không đủ dữ liệu để AI học."
        )

    # =========================================
    # DATAFRAME
    # =========================================
    with st.expander(
        "📋 Xem dữ liệu tham khảo"
    ):

        st.dataframe(
            df.head(20)
        )

else:

    st.info(
        "Hãy kiểm tra file xecu.csv."
    )
