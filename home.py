import streamlit as st
from db import cek_koneksi, hitung
from utils import terapkan_tema

terapkan_tema("home")

#NAMA = "Nama Kamu"
#NIM = "672023276"
#MATKUl = "Nama Mata Kuliah"S

st.title("Aplikasi Machine Learning + MongoDB")


st.write(
    "Aplikasi ini mendemonstrasikan regresi, klasifikasi, clustering, dan deep "
    "learning, dengan data yang tersimpan di MongoDB Atlas serta analisis "
    "komentar publik menggunakan LLM Groq."
)

st.divider()
 
# --- Status sistem ---
st.subheader("Status Sistem")
ok, pesan = cek_koneksi()
if ok:
    st.success(pesan)
else:
    st.error(pesan)
 
c1, c2, c3 = st.columns(3)
c1.metric("Komentar publik", hitung("komentar") if ok else "-")
c2.metric("Dataset tersimpan", hitung("datasets") if ok else "-")
c3.metric("Riwayat eksperimen", hitung("eksperimen") if ok else "-")
 
st.divider()

# --- Kartu fitur ---
import streamlit as st

st.subheader("Fitur")

FITUR = [
    ("Regresi", "Prediksi nilai numerik dengan 6 model, metrik MAE, RMSE, dan R².", "regresi.py", ":material/show_chart:"),
    ("Klasifikasi", "Prediksi kategori dengan 6 model; accuracy, precision, recall, F1, dan confusion matrix", "klasifikasi.py", ":material/target:"),
    ("Clustering", "3 algoritma (K-Means, Hierarchical, DBSCAN)", "clustering.py", ":material/hub:"),
    ("Komentar", "Komentar publik tersimpan di MongoDB.", "komentar.py", ":material/chat:"),
    ("AI Groq", "Tanyakan sesuatu pada AI Groq.", "ai.py", ":material/smart_toy:"),
    ("CNN", "Mengenali angka tulisan tangan (MNIST).", "cnn.py", ":material/image:"),
]

# Menampilkan fitur dalam layout 2 kolom (2x2 grid)
col1, col2 = st.columns(2)

for i, (judul, deskripsi, page_path, ikon) in enumerate(FITUR):
    target_col = col1 if i % 2 == 0 else col2
    
    with target_col:
        with st.container(border=True):
            st.markdown(f"**{judul}**")
            st.caption(deskripsi)
            st.page_link(
                page_path, 
                label=f"Coba {judul.strip()} →", 
                icon=ikon,
                use_container_width=True
            )
st.divider()

# --- Cara pakai ---
st.subheader("Cara Pakai")
st.markdown(
    "1. Pilih halaman dari menu di sebelah kiri.\n"
    "2. Pilih dataset dan model, lalu atur parameter.\n"
    "3. Klik tombol latih dan lihat hasil evaluasinya.\n"
    
)

