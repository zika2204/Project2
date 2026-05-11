import streamlit as st
import pandas as pd
import numpy as np
import scikit-learn as sklearn
from sklearn.linear_model import LinearRegression

# Cấu hình trang
st.set_page_config(page_title="AI Dự Đoán Giá Xe Máy", layout="wide")

# --- 1. HÀM TẢI VÀ XỬ LÝ DỮ LIỆU ---
@st.cache_data
def load_and_clean_data():
    try:
        # Đọc file CSV bạn đã cung cấp
        file_path = "xecu.csv"
        df = pd.read_csv(file_path)
        
        # Làm sạch tên cột
        df.columns = df.columns.str.strip()
        
        # Xử lý Price: Bỏ ngoặc kép, dấu phẩy và chuyển về số
        df['price_numeric'] = df['price'].str.replace('[",]', '', regex=True).astype(float)
        
        # Xử lý Odo: Bỏ dấu chấm (nếu có) và chuyển về số
        df['odo_numeric'] = df['odo'].astype(str).str.replace('.', '', regex=False).astype(float)
        
        # Đảm bảo các cột định dạng chữ chuẩn xác
        df['brand'] = df['brand'].astype(str).str.strip()
        df['model'] = df['model'].astype(str).str.strip()
        
        return df
    except Exception as e:
        st.error(f"Lỗi đọc dữ liệu: {e}")
        return None

# --- 2. HÀM HUẤN LUYỆN ML (LINEAR REGRESSION - GIỐNG TRONG FILE L3.IPYNB) ---
def train_model_for_bike(df, selected_model):
    # Lọc dữ liệu theo đúng Model xe người dùng chọn để tăng độ chính xác
    data_train = df[df['model'] == selected_model].copy()
    
    if len(data_train) < 3: # Cần ít nhất vài dòng để học
        return None, None
    
    # X là các cột feature: Năm sản xuất và Odo
    X = data_train[['year', 'odo_numeric']]
    # y là giá bán
    y = data_train['price_numeric']
    
    # Khởi tạo và huấn luyện mô hình giống như file L3.ipynb bạn học
    model = LinearRegression()
    model.fit(X, y)
    
    return model, data_train

# --- BẮT ĐẦU CHẠY APP ---
df = load_and_clean_data()

if df is not None:
    st.title("🏍️ AI Predictor - Dự đoán giá xe cũ")
    st.write("Mô hình được xây dựng dựa trên thuật toán **Linear Regression** (Hồi quy tuyến tính).")
    st.markdown("---")

    # --- PHẦN NHẬP LIỆU ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 Thông tin xe")
        
        # Chọn Hãng
        list_brand = sorted(df['brand'].unique())
        brand_user = st.selectbox("Chọn hãng xe:", list_brand)
        
        # Chọn Model dựa trên Hãng
        list_model = sorted(df[df['brand'] == brand_user]['model'].unique())
        model_user = st.selectbox("Chọn dòng xe (Model):", list_model)
        
        # Nhập Năm và Odo để dự đoán
        year_user = st.number_input("Năm sản xuất:", min_value=2010, max_value=2025, value=2020)
        odo_user = st.number_input("Số KM đã đi (Odo):", min_value=0, value=15000, step=1000)

    # --- PHẦN DỰ ĐOÁN ---
    with col2:
        st.subheader("💰 AI Định giá")
        
        if st.button("Dự đoán giá bán ngay", use_container_width=True):
            # Huấn luyện mô hình cho dòng xe cụ thể này
            ml_model, filtered_df = train_model_for_bike(df, model_user)
            
            if ml_model is not None:
                # Tạo mảng dữ liệu để dự đoán
                input_data = pd.DataFrame([[year_user, odo_user]], columns=['year', 'odo_numeric'])
                
                # Thực hiện dự đoán
                prediction = ml_model.predict(input_data)[0]
                
                # Tránh trường hợp dự đoán ra số âm do ít dữ liệu
                final_price = max(prediction, 0)
                
                st.metric(label="Giá dự đoán", value=f"{final_price:,.0f} VNĐ")
                
                # Hiển thị độ chính xác (R-squared) giống như lệnh model.score() trong L3.ipynb
                score = ml_model.score(filtered_df[['year', 'odo_numeric']], filtered_df['price_numeric'])
                st.caption(f"Độ tin cậy của mô hình cho dòng xe này: {score:.2%}")
                
                st.info(f"💡 AI nhận thấy: Xe đời càng cao và Odo càng thấp thì giá sẽ càng tiệm cận mức giá trần của dòng {model_user}.")
            else:
                st.error("Dữ liệu cho dòng xe này quá ít để AI có thể học và dự đoán chính xác. Vui lòng thử dòng xe khác!")

    # --- BIỂU ĐỒ ---
    st.markdown("---")
    if st.checkbox("Xem biểu đồ phân bổ giá theo năm của dòng xe này"):
        chart_data = df[df['model'] == model_user]
        st.scatter_chart(data=chart_data, x='year', y='price_numeric')

else:
    st.error("Vui lòng tải file 'datasheet AI's project - Trang tính1.csv' lên cùng thư mục code.")
