import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Cấu hình trang
st.set_page_config(page_title="AI Dự Đoán Giá Xe Máy", layout="wide")

# --- 1. HÀM TẢI VÀ XỬ LÝ DỮ LIỆU ---
@st.cache_data
def load_data():
    try:
        # Đọc file (đảm bảo tên file khớp với file bạn đã up lên GitHub)
        file_path = "datasheet AI's project - Trang tính1.csv"
        df = pd.read_csv(file_path)
        
        # Làm sạch tên cột
        df.columns = df.columns.str.strip()
        
        # Xử lý cột Price: Bỏ dấu phẩy, bỏ ngoặc kép -> Chuyển thành số
        df['price_numeric'] = df['price'].str.replace('[",]', '', regex=True).astype(float)
        
        # Xử lý cột Odo: Bỏ dấu chấm (nếu có dạng 2.500) -> Chuyển thành số
        df['odo_numeric'] = df['odo'].astype(str).str.replace('.', '', regex=False).astype(float)
        
        # Chuyển các cột định dạng chữ về string để không lỗi
        df['brand'] = df['brand'].astype(str).str.strip()
        df['model'] = df['model'].astype(str).str.strip()
        
        return df
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu: {e}")
        return None

# --- 2. HÀM HUẤN LUYỆN AI (MACHINE LEARNING) ---
@st.cache_resource
def train_ai_model(df):
    # Chọn các đặc tính để AI học
    features = ['brand', 'model', 'year', 'odo_numeric', 'condition']
    X = df[features].copy()
    y = df['price_numeric']

    # Mã hóa chữ thành số để máy học được
    le_brand = LabelEncoder()
    le_model = LabelEncoder()
    
    X['brand'] = le_brand.fit_transform(X['brand'])
    X['model'] = le_model.fit_transform(X['model'])

    # Huấn luyện mô hình Random Forest
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    
    return model, le_brand, le_model

# Chạy tải dữ liệu
df = load_data()

if df is not None:
    # Huấn luyện AI
    model, le_brand, le_model = train_ai_model(df)

    # --- GIAO DIỆN ---
    st.title("🤖 Trí Tuệ Nhân Tạo Dự Đoán Giá Xe Máy Cũ")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 Nhập thông tin xe cần định giá")
        
        # Chọn Hãng
        brand_list = sorted(df['brand'].unique())
        input_brand = st.selectbox("Chọn hãng xe:", brand_list)
        
        # Chọn Model (lọc theo hãng)
        model_list = sorted(df[df['brand'] == input_brand]['model'].unique())
        input_model = st.selectbox("Chọn dòng xe (Model):", model_list)
        
        # Nhập năm
        input_year = st.number_input("Năm sản xuất:", min_value=2000, max_value=2025, value=2020)
        
        # Nhập Odo
        input_odo = st.number_input("Số KM đã đi (Odo):", min_value=0, value=10000, step=500)
        
        # Chọn độ mới
        input_condition = st.slider("Độ mới của xe (1: Rất cũ - 10: Như mới):", 1, 10, 7)

    with col2:
        st.subheader("💰 Kết quả định giá từ AI")
        
        if st.button("BẮT ĐẦU DỰ ĐOÁN", use_container_width=True):
            # Kiểm tra xem Model này AI đã từng thấy chưa
            try:
                # Chuẩn bị dữ liệu để AI dự đoán
                brand_encoded = le_brand.transform([input_brand])[0]
                
                # Nếu người dùng nhập model mới hoàn toàn, xử lý lỗi
                try:
                    model_encoded = le_model.transform([input_model])[0]
                except:
                    model_encoded = 0
                
                input_data = np.array([[brand_encoded, model_encoded, input_year, input_odo, input_condition]])
                
                # AI đưa ra con số
                prediction = model.predict(input_data)[0]
                
                # Hiển thị kết quả
                st.metric(label="Giá dự đoán trung bình", value=f"{prediction:,.0f} VNĐ")
                
                st.success(f"Dựa trên dữ liệu, chiếc **{input_brand} {input_model}** của bạn có giá khoảng **{prediction/1000000:.1f} triệu đồng**.")
                
                # So sánh với thực tế trong file
                actual_data = df[(df['brand'] == input_brand) & (df['model'] == input_model)]
                if not actual_data.empty:
                    st.write("**📊 Tham khảo giá thực tế trong lịch sử:**")
                    st.dataframe(actual_data[['year', 'odo', 'condition', 'price']].head(5))
                
            except Exception as e:
                st.error(f"Đã xảy ra lỗi khi tính toán: {e}")

    # --- PHẦN PHÂN TÍCH THÊM ---
    st.markdown("---")
    with st.expander("🔍 Xem xu hướng giá theo năm của dòng xe này"):
        chart_data = df[df['model'] == input_model].sort_values('year')
        if not chart_data.empty:
            st.line_chart(data=chart_data, x='year', y='price_numeric')
        else:
            st.write("Không đủ dữ liệu để vẽ biểu đồ.")

else:
    st.warning("Vui lòng kiểm tra file CSV dữ liệu đầu vào.")
