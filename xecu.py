import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Cấu hình giao diện
st.set_page_config(page_title="AI Định Giá Xe Máy", layout="centered")

# --- 1. HÀM TẢI VÀ LÀM SẠCH DỮ LIỆU ---
@st.cache_data
def load_data():
    try:
        # Đọc file CSV (đảm bảo tên file này đúng với file trên GitHub của bạn)
        path = "datasheet AI's project - Trang tính1.csv"
        df = pd.read_csv(path)
        
        # Xử lý tên cột bị thừa khoảng trắng
        df.columns = df.columns.str.strip()
        
        # Chuyển Price từ chuỗi "32,000,000" thành số 32000000
        df['price_numeric'] = df['price'].str.replace('[",]', '', regex=True).astype(float)
        
        # Chuyển Odo từ "2.500" thành 2500
        df['odo_numeric'] = df['odo'].astype(str).str.replace('.', '', regex=False).astype(float)
        
        return df
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu: {e}")
        return None

df = load_data()

# --- 2. GIAO DIỆN CHÍNH ---
st.title("🏍️ Dự đoán giá xe bằng Linear Regression")
st.write("Dựa trên kiến thức Hồi quy tuyến tính từ bài học L3.")

if df is not None:
    # Sidebar để chọn dòng xe muốn huấn luyện AI
    st.sidebar.header("Cấu hình Model AI")
    all_brands = sorted(df['brand'].unique())
    selected_brand = st.sidebar.selectbox("Chọn hãng xe:", all_brands)
    
    all_models = sorted(df[df['brand'] == selected_brand]['model'].unique())
    selected_model = st.sidebar.selectbox("Chọn dòng xe để AI học:", all_models)

    # Lọc dữ liệu theo dòng xe đã chọn để train model
    data_train = df[df['model'] == selected_model].copy()

    if len(data_train) >= 3:
        # --- 3. HUẤN LUYỆN MODEL (Giống hệt file L3.ipynb) ---
        # Feature X: Năm và Odo | Target y: Giá
        X_train = data_train[['year', 'odo_numeric']]
        y_train = data_train['price_numeric']
        
        model_ai = LinearRegression()
        model_ai.fit(X=X_train, y=y_train)
        
        # --- 4. NHẬP THÔNG TIN DỰ ĐOÁN ---
        st.subheader(f"Dự đoán giá cho dòng xe: {selected_model}")
        
        col1, col2 = st.columns(2)
        with col1:
            input_year = st.number_input("Năm sản xuất:", min_value=2010, max_value=2026, value=2022)
        with col2:
            input_odo = st.number_input("Số KM đã đi (Odo):", min_value=0, value=5000, step=500)

        if st.button("Tính toán giá dự kiến"):
            # Tạo mảng để dự đoán
            X_new = pd.DataFrame([[input_year, input_odo]], columns=['year', 'odo_numeric'])
            prediction = model_ai.predict(X_new)[0]
            
            # Đảm bảo không dự đoán ra số âm
            final_price = max(prediction, 0)
            
            st.divider()
            st.success(f"### 💰 Giá AI dự đoán: {final_price:,.0f} VNĐ")
            
            # Hiển thị các chỉ số giống trong file L3.ipynb
            st.write("**Thông số kỹ thuật của mô hình:**")
            st.text(f"Hệ số chặn (Intercept): {model_ai.intercept_:,.2f}")
            st.text(f"Hệ số góc (Coefficients): {model_ai.coef_}")
            st.text(f"Độ chính xác (Score): {model_ai.score(X_train, y_train):.2%}")
            
    else:
        st.warning(f"Dòng xe {selected_model} hiện chỉ có {len(data_train)} mẫu dữ liệu. AI cần ít nhất 3 mẫu để thực hiện thuật toán Linear Regression.")

    # Hiển thị bảng dữ liệu tham khảo
    with st.expander("Xem danh sách dữ liệu gốc"):
        st.dataframe(df[['brand', 'model', 'year', 'odo', 'price']])

else:
    st.info("Hãy đảm bảo file CSV nằm đúng thư mục trên GitHub của bạn.")
