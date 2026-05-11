import streamlit as st
import pandas as pd

# Cấu hình trang
st.set_page_config(page_title="Tra cứu giá xe máy", layout="centered")

# --- DANH SÁCH DỮ LIỆU ---
hang_xe_list = [
    "Honda", "VinFast", "Yamaha", "Piaggio", "SYM", "Suzuki", "Yadea", "Pega", 
    "Dibao", "Selex", "Anbico", "Voge", "BMW", "Kawasaki", "Harley-Davidson", 
    "KTM", "Triumph", "Ducati", "Royal Enfield", "Benelli", "CFMOTO", "Lifan", "Zontes"
]

model_xe_list = [
    "Winner X Đặc biệt", "Winner X Tiêu chuẩn", "Winner V2", "SH 125i", "SH 150i", "SH 300i", 
    "SH Mode", "Air Blade", "Lead Đèn Led", "Lead", "Vision", "Vario 125", "Vario 150", 
    "Vario 160", "PCX 125", "PCX 150", "PCX 160", "Future 125 FI", "Future 125", 
    "Wave Alpha", "Wave RSX", "Super Cub C125", "Monkey", "Zoomer X", "Dylan", "Beat", 
    "Sonic 150", "CBR 150R", "Rebel 500", "CB300R", "CB500X", "ADV 150", "ADV 160", 
    "Exciter 150", "Exciter 155 VVA", "Grand Filano", "NVX 155", "Janus", "Sirius", 
    "Jupiter RC", "Jupiter Finn", "R15 v3", "YZF-R15", "YZF-R7", "YZF-R6", "MT-15", 
    "MT-07", "MT-09", "FZ150i", "PG-1", "XS 125", "XMAX 250", "XMAX 300", "Lexi 125", 
    "FreeGo S", "Acruzo", "Mio 125", "Tracer 9 GT", "Vespa Primavera 125", "Vespa Primavera 150", 
    "Vespa Sprint 125", "Vespa Sprint S 150", "Vespa LX 125", "Vespa GTS 300", 
    "Vespa GTS 300 Super", "Vespa 946", "Liberty 125", "Medley 150", "Zip 50", "Fly 125", 
    "Beverly 300", "Raider 150", "Satria F150", "Viva 115", "Address 125", "Hayate 125", 
    "Impulse", "Skydrive", "Burgman 200", "GSX-S150", "GSX-R150", "V-Strom 250", 
    "V-Strom 650", "Attila V", "Shark 50", "Elite 50", "Elite 150", "Galaxy 125", 
    "Angel 125", "Elegant 50", "VF 185", "Happy", "Simply 125", "MaxSym 400", 
    "JET 14 150", "JET X 150", "ADX 125", "TL 500", "Klara S (2022)", "Klara S", 
    "Vento S", "Theon S", "Feliz S", "Feliz Neo", "Evo 200", "Impes", "Tempest", 
    "E3 Lite", "G5", "Xmen", "XMEN", "Buya", "Buya E", "A10", "Lado", "L1", "S1", 
    "C2", "Kiki", "Pega Camel 1", "Ninja 400", "Z900", "Versys 650", "KLX 230", 
    "G 310 R", "G 310 GS", "F 750 GS", "Iron 883", "Forty Eight", "Street Bob", 
    "Street Twin", "Trident 660", "Scrambler Icon", "Monster 821", "Classic 350", 
    "Meteor 350", "TNT 150i", "Leoncino 250", "Benelli", "300AC", "300SR", "650MT", 
    "KP150", "ZT 155"
]

years = list(range(2014, 2025))

# --- GIAO DIỆN ---
st.title("🏍️ Hệ thống tra cứu giá xe")
st.write("Vui lòng chọn thông tin xe để xem giá dự kiến.")

# Sử dụng cột để giao diện đẹp hơn
col1, col2 = st.columns(2)

with col1:
    hang_chon = st.selectbox("Chọn hãng xe:", hang_xe_list)
    nam_chon = st.selectbox("Chọn năm sản xuất:", years)

with col2:
    model_chon = st.selectbox("Chọn Model xe:", model_xe_list)

# Nút bấm tính toán
if st.button("Xem giá xe", use_container_width=True):
    # Logic hiển thị giá (Sau này bạn sẽ thay bằng việc đọc file CSV/Excel)
    st.divider()
    st.subheader(f"Kết quả cho: {hang_chon} {model_chon} ({nam_chon})")
    
    # Placeholder: Hiện tại mình để tạm một con số giả lập
    # Khi bạn có file, mình sẽ viết code tìm kiếm trong file dựa trên hang_chon, model_chon, nam_chon
    gia_tam_tinh = "Đang chờ kết nối dữ liệu..." 
    
    st.info(f"**Giá đề xuất:** {gia_tam_tinh}")
    st.warning("Lưu ý: Giá trên chỉ mang tính chất tham khảo dựa trên dữ liệu thị trường.")

st.sidebar.header("Cấu hình dữ liệu")
uploaded_file = st.sidebar.file_uploader("")
