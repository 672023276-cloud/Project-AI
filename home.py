import streamlit as st
from db import cek_koneksi, hitung
'''
#
NAMA = "Nama Kamu"
NIM = "672023276"
MATKUL = "Nama Mata Kuliah"
'''
st.title("Aplikasi Machine Learning + MongoDB")
# st.caption(f"{NAMA} · {NIM} · {MATKUL} · Tes Akhir Semester")

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
    ("Regresi", "Prediksi nilai numerik dengan 5 model, metrik MSE, RMSE, dan R².", "regresi.py"),
    ("Klasifikasi", "Prediksi kategori dengan 5 model, confusion matrix, dan F1.", "klasifikasi.py"),
    ("Clustering", "K-Means dengan Elbow Method, Silhouette Score, dan PCA 2D.", "clustering.py"),
    ("Komentar & AI", "Komentar publik tersimpan di MongoDB, diringkas dan dinilai sentimennya oleh Groq.", "komentar.py"),
]
kolom = st.columns(2)
for i, (judul, desk, path) in enumerate(FITUR):
    with kolom[i % 2].container(border=True):
        st.markdown(f"**{judul}**")
        st.write(desk)
        try:
            st.page_link(path, label=f"Buka {judul}")
        except Exception:
            st.caption("Halaman belum dibuat")

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

'''
import sklearn
from sklearn.datasets import load_breast_cancer
import pandas as pd
import streamlit as st

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = pd.Series(data.target).map(dict(enumerate(data.target_names)))

tab1, tab2, tab3 = st.tabs(["Tentang Data", "Tabel & Statistik", "Visualisasi"])

with tab1:
    st.text(data.DESCR)

with tab2:
    st.dataframe(df.head(20), use_container_width=True)
    st.dataframe(df.describe().T, use_container_width=True)

with tab3:
    st.bar_chart(df["target"].value_counts())
    fitur = st.selectbox("Pilih fitur", data.feature_names)
    st.bar_chart(df[fitur].value_counts(bins=20).sort_index())
    '''