import streamlit as st
import pandas as pd

# Cấu hình trang
st.set_page_config(page_title="Hệ thống tra cứu giá xe máy", layout="centered")

# --- HÀM TẢI DỮ LIỆU ---
@st.cache_data
def load_data():
    try:
        # Thay đổi tên file này cho đúng với file của bạn trên GitHub
        file_path = "datasheet AI's project - Trang tính1.csv"
        df = pd.read_csv(file_path)
        
        # Làm sạch tên cột (xóa khoảng trắng thừa)
        df.columns = df.columns.str.strip()
        
        # Chuyển đổi năm về kiểu số nguyên
        df['year'] = df['year'].astype(int)
        
        # Chuyển đổi odo về kiểu chuỗi để hiển thị đẹp hơn
        df['odo'] = df['odo'].astype(str)
        
        return df
    except Exception as e:
        st.error(f"Không tìm thấy hoặc không thể đọc file dữ liệu: {e}")
        return None

# Load dữ liệu
df = load_data()

# --- GIAO DIỆN APP ---
st.title("🏍️ Hệ thống tra cứu giá xe")
st.write("Dữ liệu được cập nhật tự động từ file datasheet.")

if df is not None:
    # --- PHẦN CHỌN DỮ LIỆU ---
    # 1. Chọn Hãng (Lấy từ cột 'brand')
    list_hang = sorted(df['brand'].unique())
    hang_chon = st.selectbox("Chọn hãng xe:", list_hang)

    # 2. Chọn Model (Chỉ hiện các model thuộc hãng đã chọn)
    df_filtered_by_brand = df[df['brand'] == hang_chon]
    list_model = sorted(df_filtered_by_brand['model'].unique())
    model_chon = st.selectbox("Chọn model xe:", list_model)

    # 3. Chọn Năm (Chỉ hiện các năm có sẵn của model đó)
    df_filtered_by_model = df_filtered_by_brand[df_filtered_by_brand['model'] == model_chon]
    list_year = sorted(df_filtered_by_model['year'].unique(), reverse=True)
    nam_chon = st.selectbox("Chọn năm sản xuất:", list_year)

    # --- HIỂN THỊ KẾT QUẢ ---
    if st.button("Xem giá xe", use_container_width=True):
        # Tìm dòng dữ liệu khớp chính xác
        ket_qua = df_filtered_by_model[df_filtered_by_model['year'] == nam_chon]
        
        st.divider()
        
        if not ket_qua.empty:
            # Vì có thể một model/năm có nhiều dòng, ta lấy dòng đầu tiên
            xe = ket_qua.iloc[0]
            
            st.subheader(f"Kết quả: {hang_chon} {model_chon}")
            
            # Hiển thị giá nổi bật
            st.success(f"### 💰 Giá đề xuất: {xe['price']} VNĐ")
            
            # Hiển thị các thông số khác trong 2 cột
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**📍 Khu vực:** {xe['location']}")
                st.write(f"**🛣️ Odo:** {xe['odo']} km")
            with c2:
                st.write(f"**🛠️ Tình trạng sửa chữa:** {xe['repaired_parts']}")
                st.write(f"**⭐ Độ mới:** {xe['condition']}/10")
            
            st.warning("⚠️ Lưu ý: Giá trên chỉ mang tính chất tham khảo tại thời điểm tra cứu.")
        else:
            st.error("Rất tiếc, không tìm thấy dữ liệu cho lựa chọn này.")

else:
    st.info("Vui lòng kiểm tra file CSV và đảm bảo file nằm cùng thư mục với app.py")

# --- SIDEBAR (Tùy chọn thêm) ---
st.sidebar.header("Thông tin")
st.sidebar.write("Hệ thống tra cứu giá xe máy cũ/mới dựa trên dữ liệu thị trường.")
