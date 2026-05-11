import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# =========================
# CẤU HÌNH TRANG
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
        df = pd.read_csv("xecu.csv")

        # Chuẩn hóa tên cột
        df.columns = df.columns.str.strip().str.lower()

        # =========================
        # XỬ LÝ GIÁ
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

        # Xóa dữ liệu trống
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
st.write(
    "Ứng dụng sử dụng Linear Regression để dự đoán giá xe máy cũ."
)

# =========================
# NẾU LOAD DATA THÀNH CÔNG
# =========================
if df is not None:

    # =========================
    # SIDEBAR
    # =========================
    st.sidebar.header("⚙️ Chọn xe")

    # Danh sách hãng
    all_brands = sorted(df["brand"].unique())

    selected_brand = st.sidebar.selectbox(
        "Hãng xe",
        all_brands
    )

    # Danh sách model theo hãng
    all_models = sorted(
        df[df["brand"] == selected_brand]["model"].unique()
    )

    selected_model = st.sidebar.selectbox(
        "Dòng xe",
        all_models
    )

    # =========================
    # DATA TRAIN
    # =========================

    # Data cùng model
    data_same_model = df[
        df["model"] == selected_model
    ]

    # Data cùng hãng
    data_same_brand = df[
        df["brand"] == selected_brand
    ]

    # Gộp data
    data_train = pd.concat([
        data_same_model,
        data_same_brand
    ])

    # Xóa trùng
    data_train = data_train.drop_duplicates()

    # =========================
    # CHECK DATA
    # =========================
    if len(data_train) >= 5:

        st.write(
            f"📊 AI đang học từ {len(data_train)} mẫu dữ liệu"
        )

        # =========================
        # FEATURE & TARGET
        # =========================
        X_train = data_train[
            ["year", "odo_numeric"]
        ]

        y_train = data_train[
            "price_numeric"
        ]

        # =========================
        # TRAIN MODEL
        # =========================
        model_ai = LinearRegression()

        model_ai.fit(X_train, y_train)

        # =========================
        # NHẬP THÔNG TIN XE
        # =========================
        st.subheader(
            f"🔍 Dự đoán giá xe {selected_model}"
        )

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

        # =========================
        # BUTTON DỰ ĐOÁN
        # =========================
        if st.button("💰 Dự đoán giá"):

            # Tạo dữ liệu mới
            X_new = pd.DataFrame(
                [[input_year, input_odo]],
                columns=[
                    "year",
                    "odo_numeric"
                ]
            )

            # Predict
            prediction = model_ai.predict(X_new)[0]

            # Không cho âm
            final_price = max(prediction, 0)

            st.divider()

            # =========================
            # KẾT QUẢ
            # =========================
            st.success(
                f"### 💵 Giá dự đoán: {final_price:,.0f} VNĐ"
            )

            # =========================
            # THÔNG SỐ AI
            # =========================
            st.subheader("📈 Thông số mô hình")

            year_coef = model_ai.coef_[0]
            odo_coef = model_ai.coef_[1]

            st.write(
                f"**Intercept:** {model_ai.intercept_:,.2f}"
            )

            st.write(
                f"**Hệ số năm sản xuất:** {year_coef:,.2f}"
            )

            st.write(
                f"**Hệ số ODO:** {odo_coef:,.2f}"
            )

            # Logic ODO
            if odo_coef > 0:
                st.warning(
                    "⚠️ AI đang học chưa chính xác "
                    "(ODO tăng nhưng giá cũng tăng). "
                    "Cần thêm dữ liệu thực tế."
                )
            else:
                st.success(
                    "✅ AI đã học đúng logic thị trường."
                )

            # Score
            score = model_ai.score(
                X_train,
                y_train
            )

            st.write(
                f"**Độ chính xác (R² Score):** {score:.2%}"
            )

        # =========================
        # XEM DATA GỐC
        # =========================
        with st.expander("📋 Xem dữ liệu gốc"):

            st.dataframe(
                data_train[
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
        st.warning(
            "❌ Không đủ dữ liệu để train AI."
        )

else:
    st.info(
        "Hãy kiểm tra file xecu.csv."
    )
