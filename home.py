import streamlit as st
from db import cek_koneksi, hitung


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
st.subheader("Fitur")
FITUR = [
    ("Regresi", "Prediksi nilai numerik dengan 5 model, metrik MSE, RMSE, dan R²."),
    ("Klasifikasi", "Prediksi kategori dengan 5 model, confusion matrix, dan F1."),
    ("Clustering", "K-Means dengan Elbow Method, Silhouette Score, dan PCA 2D."),
    ("Komentar & AI", "Komentar publik tersimpan di MongoDB, diringkas dan dinilai sentimennya oleh Groq."),
]

kolom = st.columns(2)
for i, (judul, desk) in enumerate(FITUR):
    with kolom[i % 2].container(border=True):
        st.markdown(f"**{judul}**")
        st.write(desk)
st.divider()

# --- Cara pakai ---
st.subheader("Cara Pakai")
st.markdown(
    "1. Pilih halaman dari menu di sebelah kiri.\n"
    "2. Pilih dataset dan model, lalu atur parameter.\n"
    "3. Klik tombol latih dan lihat hasil evaluasinya.\n"
    "4. Tambahkan dataset atau komentar, datanya tersimpan di MongoDB."
)

# --- Teknologi ---
st.subheader("Teknologi")
st.write("Streamlit · scikit-learn · TensorFlow · MongoDB Atlas · Groq · Pandas · Matplotlib")
