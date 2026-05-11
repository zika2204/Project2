import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# =========================
# CONFIG GIAO DIỆN
# =========================
st.set_page_config(
    page_title="AI Định Giá Xe Máy",
    page_icon="🏍️",
    layout="centered"
)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    try:
        # Đọc file CSV
        path = "xecu.csv"
        df = pd.read_csv(path)

        # Làm sạch tên cột
        df.columns = df.columns.str.strip().str.lower()

        # =========================
        # XỬ LÝ PRICE
        # =========================
        df["price_numeric"] = (
            df["price"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace('"', "", regex=False)
            .astype(float)
        )

        # =========================
        # XỬ LÝ ODO
        # =========================
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

# =========================
# GIAO DIỆN CHÍNH
# =========================
st.title("🏍️ AI Dự Đoán Giá Xe Máy")
st.write("Ứng dụng sử dụng thuật toán Linear Regression để dự đoán giá xe.")

if df is not None:

    # =========================
    # SIDEBAR
    # =========================
    st.sidebar.header("⚙️ Cấu hình AI")

    # Chọn hãng xe
    all_brands = sorted(df["brand"].unique())
    selected_brand = st.sidebar.selectbox(
        "Chọn hãng xe:",
        all_brands
    )

    # =========================
    # TRAIN MODEL
    # =========================
    # Train theo hãng để có nhiều data hơn
    data_train = df[df["brand"] == selected_brand].copy()

    st.write(f"📊 Số lượng dữ liệu huấn luyện: {len(data_train)} mẫu")

    if len(data_train) >= 5:

        # Features
        X_train = data_train[["year", "odo_numeric"]]

        # Target
        y_train = data_train["price_numeric"]

        # Tạo model
        model_ai = LinearRegression()

        # Huấn luyện
        model_ai.fit(X_train, y_train)

        # =========================
        # NHẬP DỮ LIỆU
        # =========================
        st.subheader("🔍 Nhập thông tin xe")

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
                "Số KM đã đi",
                min_value=0,
                value=5000,
                step=500
            )

        # =========================
        # DỰ ĐOÁN
        # =========================
        if st.button("💰 Dự đoán giá"):

            X_new = pd.DataFrame(
                [[input_year, input_odo]],
                columns=["year", "odo_numeric"]
            )

            prediction = model_ai.predict(X_new)[0]

            # Không cho giá âm
            final_price = max(prediction, 0)

            st.divider()

            st.success(
                f"### 💵 Giá dự đoán: {final_price:,.0f} VNĐ"
            )

            # =========================
            # THÔNG SỐ MODEL
            # =========================
            st.subheader("📈 Thông số AI")

            year_coef = model_ai.coef_[0]
            odo_coef = model_ai.coef_[1]

            st.write(f"**Intercept:** {model_ai.intercept_:,.2f}")

            st.write(
                f"**Hệ số năm sản xuất:** {year_coef:,.2f}"
            )

            st.write(
                f"**Hệ số ODO:** {odo_coef:,.2f}"
            )

            # Kiểm tra logic ODO
            if odo_coef > 0:
                st.warning(
                    "⚠️ AI đang học rằng số KM tăng thì giá tăng. "
                    "Điều này cho thấy dữ liệu huấn luyện chưa đủ tốt."
                )
            else:
                st.success(
                    "✅ AI đã học đúng logic: xe chạy nhiều sẽ mất giá."
                )

            # Score
            score = model_ai.score(X_train, y_train)

            st.write(f"**Độ chính xác (R² Score):** {score:.2%}")

        # =========================
        # HIỂN THỊ DỮ LIỆU
        # =========================
        with st.expander("📋 Xem dữ liệu gốc"):
            st.dataframe(
                data_train[
                    ["brand", "model", "year", "odo", "price"]
                ]
            )

    else:
        st.warning(
            "❌ Không đủ dữ liệu để huấn luyện AI. "
            "Cần ít nhất 5 mẫu dữ liệu."
        )

else:
    st.info(
        "Hãy đảm bảo file dataset.csv nằm đúng thư mục project."
    )
