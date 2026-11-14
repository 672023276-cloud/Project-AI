import streamlit as st
from db import cek_koneksi, hitung
from utils import terapkan_tema

terapkan_tema("home")



st.title("Aplikasi Machine Learning + MongoDB")


st.write(
    "Aplikasi ini mendemonstrasikan regresi, klasifikasi, clustering, dan Deep Learning CNN "
    "untuk analisis dan prediksi data, serta pengambilan data real-time dari Open-Meteo. "
    "Dilengkapi MongoDB Atlas untuk penyimpanan data, AI Groq untuk interaksi, "
    "dan analisis sentimen otomatis terhadap komentar pengguna."
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
    # (judul, deskripsi, file halaman, ikon)
    ("Regresi", "Prediksi nilai numerik dengan 6 model, metrik MAE, RMSE, dan R².",
     "regresi.py", ":material/show_chart:"),
    ("Klasifikasi", "Prediksi kategori dengan 6 model; accuracy, precision, recall, F1, dan confusion matrix.",
     "klasifikasi.py", ":material/target:"),
    ("Clustering", "3 algoritma (K-Means, Hierarchical, DBSCAN).",
     "clustering.py", ":material/hub:"),
    ("Komentar", "Komentar publik tersimpan di MongoDB.",
     "komentar.py", ":material/chat:"),
    ("AI Groq", "Tanyakan sesuatu pada AI Groq.",
     "ai.py", ":material/smart_toy:"),
    ("CNN", "Mengenali angka tulisan tangan (MNIST).",
     "CNN.py", ":material/auto_awesome:"),
    ("Sentimen", "Analisis sentimen otomatis menggunakan LLM Groq.",
     "sentiment.py", ":material/psychology:"),
    ("Realtime", "Mengambil data cuaca secara real-time dari Open-Meteo dan menyimpannya di MongoDB.", 
     "realtime.py", ":material/sensors:"),
]


for i in range(0, len(FITUR), 2):
    baris = st.columns(2)
    for kolom, (judul, isi, halaman, ikon) in zip(baris, FITUR[i:i + 2]):
        with kolom:
            with st.container(border=True, height=190):
                st.markdown(f"**{judul.strip()}**")
                st.write(isi)
                st.page_link(halaman, label=f"Coba {judul.strip()}", icon=ikon)

st.divider()

# --- Cara pakai ---
st.subheader("Cara Pakai")
st.markdown(
    "1. Pilih halaman dari menu di sebelah kiri.\n"
    "2. Pilih dataset dan model, lalu atur parameter.\n"
    "3. Klik tombol latih dan lihat hasil evaluasinya.\n"
    
)

