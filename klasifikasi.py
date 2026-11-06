import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timezone
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

UCI_CLASSIFICATION_DATASETS = {
    "Breast Cancer Wisconsin": 17,
    "Heart Disease": 45,
    "Wine": 109,
    "Iris": 53,
}

MODEL_NAMES = [
    "Logistic Regression", "KNN", "Decision Tree",
    "Random Forest", "SVM", "Naive Bayes",
]


def make_model(nama, random_state):
    model = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "SVM": SVC(kernel="rbf", random_state=random_state),
        "Naive Bayes": GaussianNB(),
    }[nama]
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def preprocess_fitur(X):
    return pd.get_dummies(X.copy(), drop_first=True).astype(float)


def siapkan_target(y_raw, nama_dataset):
    y = y_raw.iloc[:, 0]
    if nama_dataset == "Heart Disease":
        # Nilai 0 = sehat, 1-4 = ada penyakit; disederhanakan jadi dua kelas
        y = (pd.to_numeric(y, errors="coerce").fillna(0) > 0).astype(int)
        return y.values, ["Sehat", "Penyakit jantung"]
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    return y_enc, [str(c) for c in le.classes_]


def bagi_data(X, y, test_size, random_state):
    stratify = y if np.bincount(y).min() >= 2 else None
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )


def evaluasi(nama, X_train, X_test, y_train, y_test, random_state):
    pipe = make_model(nama, random_state)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    return pipe, pred, {
        "Model": nama,
        "Accuracy": float(accuracy_score(y_test, pred)),
        "Precision": float(precision_score(y_test, pred, average="weighted", zero_division=0)),
        "Recall": float(recall_score(y_test, pred, average="weighted", zero_division=0)),
        "F1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
    }


def pentingnya_fitur(pipe, kolom):
    m = pipe.named_steps["model"]
    if hasattr(m, "feature_importances_"):
        nilai = m.feature_importances_
    elif hasattr(m, "coef_"):
        nilai = np.abs(m.coef_)
        nilai = nilai.mean(axis=0) if nilai.ndim > 1 else nilai
    else:
        return None
    return pd.Series(nilai, index=kolom).sort_values(ascending=False)


def simpan_eksperimen(dokumen):
    from db import get_db
    dokumen["waktu"] = datetime.now(timezone.utc)
    get_db()["eksperimen"].insert_one(dokumen)


# ===== UI =====
st.title("Klasifikasi")
st.caption("Prediksi kategori menggunakan dataset UCI Machine Learning Repository.")


@st.cache_data(show_spinner="Mengunduh dataset dari UCI...")
def load_uci_data(dataset_id):
    repo = fetch_ucirepo(id=dataset_id)
    return repo.data.features, repo.data.targets


with st.sidebar:
    st.header("Pengaturan")
    pilihan_dataset = st.selectbox("Dataset UCI", list(UCI_CLASSIFICATION_DATASETS.keys()))
    test_size = st.slider("Test size", 0.1, 0.5, 0.2, step=0.05)
    random_state = int(st.number_input("Random state", 0, 1000, 42, step=1))

try:
    X_raw, y_raw = load_uci_data(UCI_CLASSIFICATION_DATASETS[pilihan_dataset])
except Exception as e:
    st.error(f"Gagal mengunduh dataset: {e}")
    st.stop()

X = preprocess_fitur(X_raw)
y, kelas = siapkan_target(y_raw, pilihan_dataset)
X_train, X_test, y_train, y_test = bagi_data(X, y, test_size, random_state)

tab_data, tab_latih, tab_banding = st.tabs(["Data", "Latih Model", "Bandingkan Semua"])

with tab_data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah baris", X.shape[0])
    c2.metric("Jumlah fitur", X.shape[1])
    c3.metric("Jumlah kelas", len(kelas))
    c4.metric("Nilai kosong", int(X_raw.isnull().sum().sum()))
    st.write("Distribusi kelas (data seimbang atau tidak)")
    dist = pd.Series(y).map(dict(enumerate(kelas))).value_counts()
    st.bar_chart(dist)
    st.write("Cuplikan data")
    st.dataframe(X_raw.head(20), use_container_width=True)
    st.write("Statistik deskriptif")
    st.dataframe(X_raw.describe().T, use_container_width=True)

with tab_latih:
    pilihan_model = st.selectbox("Model", MODEL_NAMES)
    if st.button("Latih model", type="primary"):
        with st.spinner("Melatih model..."):
            pipe, pred, hasil = evaluasi(
                pilihan_model, X_train, X_test, y_train, y_test, random_state
            )
        st.session_state["kls_latih"] = {
            "dataset": pilihan_dataset, "pipe": pipe, "pred": pred, "hasil": hasil,
            "y_test": np.asarray(y_test), "kolom": list(X.columns), "kelas": kelas,
        }

    r = st.session_state.get("kls_latih")
    if r and r["dataset"] == pilihan_dataset:
        h = r["hasil"]
        st.success(f"Hasil {h['Model']} pada {pilihan_dataset}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{h['Accuracy']:.4f}")
        m2.metric("Precision", f"{h['Precision']:.4f}")
        m3.metric("Recall", f"{h['Recall']:.4f}")
        m4.metric("F1-score", f"{h['F1']:.4f}")

        cm = confusion_matrix(r["y_test"], r["pred"], labels=range(len(r["kelas"])))
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=r["kelas"], yticklabels=r["kelas"])
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
        plt.close(fig)

        laporan = classification_report(
            r["y_test"], r["pred"], labels=range(len(r["kelas"])),
            target_names=r["kelas"], output_dict=True, zero_division=0,
        )
        st.write("Laporan per kelas")
        st.dataframe(pd.DataFrame(laporan).T.round(4), use_container_width=True)

        penting = pentingnya_fitur(r["pipe"], r["kolom"])
        if penting is not None:
            st.write("Fitur paling berpengaruh")
            st.bar_chart(penting.head(10))
        else:
            st.caption("Model ini tidak menyediakan feature importance.")

        if st.button("Simpan hasil ke MongoDB", key="simpan_kls"):
            try:
                simpan_eksperimen({
                    "jenis": "klasifikasi", "dataset": pilihan_dataset,
                    "model": h["Model"], "test_size": test_size,
                    "random_state": random_state,
                    "metrik": {k: round(v, 4) for k, v in h.items() if k != "Model"},
                })
                st.success("Hasil tersimpan di MongoDB.")
            except Exception as e:
                st.error(f"Gagal menyimpan: {e}")

with tab_banding:
    st.write("Latih semua model dengan pengaturan yang sama, lalu bandingkan.")
    if st.button("Bandingkan semua model"):
        baris = []
        bar = st.progress(0.0)
        for i, nama in enumerate(MODEL_NAMES):
            _, _, h = evaluasi(nama, X_train, X_test, y_train, y_test, random_state)
            baris.append(h)
            bar.progress((i + 1) / len(MODEL_NAMES))
        st.session_state["kls_banding"] = {
            "dataset": pilihan_dataset,
            "tabel": pd.DataFrame(baris).sort_values("F1", ascending=False).reset_index(drop=True),
        }

    b = st.session_state.get("kls_banding")
    if b and b["dataset"] == pilihan_dataset:
        tabel = b["tabel"]
        st.dataframe(tabel.round(4), use_container_width=True)
        st.bar_chart(tabel.set_index("Model")["F1"])
        terbaik = tabel.iloc[0]
        st.info(
            f"Model terbaik: {terbaik['Model']} dengan F1-score {terbaik['F1']:.3f} "
            f"dan accuracy {terbaik['Accuracy']:.3f}. F1 dipakai sebagai acuan utama "
            f"karena memperhitungkan precision dan recall sekaligus."
        )