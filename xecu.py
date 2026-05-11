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
    font-size: 48px;
    font-weight: bold;
    color: #E53935;
}

.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
    font-size: 18px;
}

.result-box {
    background-color: #f5f5f5;
    padding: 25px;
    border-radius: 15px;
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

        df = pd.read_csv("xecu.csv")

        # Chuẩn hóa tên cột
        df.columns = df.columns.str.strip().str.lower()

        # =====================================
        # XỬ LÝ PRICE
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

        # =====================================
        # THÊM CỘT CONDITION
        # =====================================
        # Nếu file CSV chưa có condition
        if "condition" not in df.columns:
            df["condition"] = 8

        # =====================================
        # THÊM CỘT PARTS_CHANGED
        # =====================================
        if "parts_changed" not in df.columns:
            df["parts_changed"] = 0

        return df.dropna()

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
    '<p class="subtitle">AI dự đoán giá xe máy cũ bằng Machine Learning</p>',
    unsafe_allow_html=True
)

# =====================================
# MAIN APP
# =====================================
if df is not None:

    st.subheader("📌 Nhập thông tin xe")

    # =====================================
    # BRAND
    # =====================================
    all_brands = sorted(df["brand"].unique())

    selected_brand = st.selectbox(
        "Hãng xe",
        all_brands
    )

    # =====================================
    # MODEL
    # =====================================
    all_models = sorted(
        df[df["brand"] == selected_brand]["model"].unique()
    )

    selected_model = st.selectbox(
        "Dòng xe",
        all_models
    )

    # =====================================
    # YEAR & ODO
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
    # CONDITION
    # =====================================
    input_condition = st.slider(
        "Tình trạng xe (0 - 10)",
        min_value=0,
        max_value=10,
        value=8
    )

    # =====================================
    # PARTS CHANGED
    # =====================================
    parts_changed = st.radio(
        "Xe đã thay phụ tùng chưa?",
        ["Chưa", "Đã thay"]
    )

    # Chuyển thành số
    parts_value = 1 if parts_changed == "Đã thay" else 0

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

        # Features
        X_train = data_train[
            [
                "year",
                "odo_numeric",
                "condition",
                "parts_changed"
            ]
        ]

        # Target
        y_train = data_train[
            "price_numeric"
        ]

        # Model
        model_ai = LinearRegression()

        model_ai.fit(X_train, y_train)

        # =====================================
        # BUTTON
        # =====================================
        if st.button("💰 Dự đoán giá"):

            # Input mới
            X_new = pd.DataFrame(
                [[
                    input_year,
                    input_odo,
                    input_condition,
                    parts_value
                ]],
                columns=[
                    "year",
                    "odo_numeric",
                    "condition",
                    "parts_changed"
                ]
            )

            # Predict
            prediction = model_ai.predict(X_new)[0]

            final_price = max(prediction, 0)

            # =====================================
            # KHẤU HAO
            # =====================================
            newest_year = data_train["year"].max()

            years_used = newest_year - input_year

            depreciation_year = (
                years_used * 0.05 * final_price
            )

            depreciation_odo = (
                input_odo / 1000
            ) * 120000

            # Condition ảnh hưởng giá
            condition_loss = (
                (10 - input_condition)
                * 0.03
                * final_price
            )

            # Parts changed ảnh hưởng giá
            parts_loss = (
                0.05 * final_price
                if parts_value == 1
                else 0
            )

            total_depreciation = (
                depreciation_year
                + depreciation_odo
                + condition_loss
                + parts_loss
            )

            # =====================================
            # HIỂN THỊ KẾT QUẢ
            # =====================================
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
    Giá được AI dự đoán dựa trên:
    năm sản xuất, số KM đã chạy,
    tình trạng xe và phụ tùng.
    </p>

    </div>
    """,
    unsafe_allow_html=True
)
            )

    else:
        st.warning(
            "Không đủ dữ liệu để AI học."
        )

    # =====================================
    # DATAFRAME
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
