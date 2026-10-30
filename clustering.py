import streamlit as st
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Import berbagai algoritma klasifikasi dari Scikit-Learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

st.title("Eksplorasi Dataset & Algoritma Klasifikasi")

# 1. Pengguna memilih Dataset
DATASET_MAP = {
    "Iris": 53,
    "Heart Disease": 45,
    "Wine": 109,
    "Breast Cancer": 17
}
pilihan_dataset = st.selectbox("1. Pilih Dataset:", list(DATASET_MAP.keys()))

# 2. Pengguna memilih Algoritma Klasifikasi
MODEL_MAP = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
    "Support Vector Machine (SVM)": SVC(),
    "Naive Bayes": GaussianNB()
}
pilihan_model = st.selectbox("2. Pilih Algoritma Klasifikasi:", list(MODEL_MAP.keys()))

# Fungsi Load Data
@st.cache_data
def load_data(dataset_id):
    dataset = fetch_ucirepo(id=dataset_id)
    return dataset.data.features, dataset.data.targets

X, y = load_data(DATASET_MAP[pilihan_dataset])

# 3. Eksekusi Klasifikasi
if st.button("Jalankan Model"):
    # Pra-pemrosesan sederhana (Membersihkan angka & NaN)
    X_clean = X.select_dtypes(include=['number']).fillna(X.mean(numeric_only=True))
    y_clean = y.values.ravel() if isinstance(y, pd.DataFrame) else y

    # Pembagian Data
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42
    )

    # Mengambil objek model berdasarkan pilihan pengguna
    model = MODEL_MAP[pilihan_model]

    # Training & Prediksi
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Menampilkan Hasil
    akurasi = accuracy_score(y_test, y_pred)
    st.success(f"Hasil Akurasi **{pilihan_model}** pada Dataset **{pilihan_dataset}**: **{akurasi * 100:.2f}%**")