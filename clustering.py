import streamlit as st
from data_tambahan import (
    muat_dataset, muat_tambahan, daftar_unggahan, render_form, render_unggah_dataset,
    catat_eksperimen, tampilkan_perbandingan,
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from ucimlrepo import fetch_ucirepo
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, dendrogram

from utils import terapkan_tema
terapkan_tema("clustering")

UCI_CLUSTERING_DATASETS = {
    "Iris": 53,
    "Wine": 109,
    "Seeds": 236,
    "Wholesale Customers": 292,
}

ALGORITMA = ["K-Means", "Hierarchical", "DBSCAN"]
PALET = ["#4DA3FF", "#2ED3A0", "#FF8A5B", "#B18CFF", "#FFD166",
         "#6EE7F9", "#F472B6", "#A3E635", "#F87171", "#94A3B8"]


def siapkan_fitur(X_raw):
    X = pd.get_dummies(X_raw.copy(), drop_first=True).astype(float)
    pra = Pipeline([("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())])
    return X, pra.fit_transform(X)


def jalankan(nama, Xs, k, linkage_method, eps, min_samples, random_state):
    if nama == "K-Means":
        return KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(Xs)
    if nama == "Hierarchical":
        return AgglomerativeClustering(n_clusters=k, linkage=linkage_method).fit_predict(Xs)
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Xs)


def skor(Xs, labels):
    """Silhouette dan Davies-Bouldin; titik noise DBSCAN (-1) diabaikan."""
    mask = labels != -1
    n_cluster = len(set(labels[mask]))
    if n_cluster < 2 or mask.sum() <= n_cluster:
        return None, None
    return (float(silhouette_score(Xs[mask], labels[mask])),
            float(davies_bouldin_score(Xs[mask], labels[mask])))


def elbow(Xs, k_min, k_max, random_state):
    ks = list(range(k_min, k_max + 1))
    inertia, sil = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(Xs)
        inertia.append(float(km.inertia_))
        sil.append(float(silhouette_score(Xs, km.labels_)))
    return ks, inertia, sil


def ringkas(nama, Xs, labels):
    s, d = skor(Xs, labels)
    return {
        "Algoritma": nama,
        "Jumlah cluster": int(len(set(labels)) - (1 if -1 in labels else 0)),
        "Titik noise": int((labels == -1).sum()),
        "Silhouette": s,
        "Davies-Bouldin": d,
    }


def simpan_eksperimen(dokumen):
    from db import get_db
    dokumen["waktu"] = datetime.now(timezone.utc)
    get_db()["eksperimen"].insert_one(dokumen)



def retrain_clustering(algoritma, X_raw, baru, k, metode_link, eps, min_samples, random_state):
    """Kelompokkan data UCI saja (sebelum) dan UCI + tambahan (sesudah)."""
    _, Xs_d = siapkan_fitur(X_raw[~baru])
    lab_d = jalankan(algoritma, Xs_d, k, metode_link, eps, min_samples, random_state)
    _, Xs_g = siapkan_fitur(X_raw)
    lab_g = jalankan(algoritma, Xs_g, k, metode_link, eps, min_samples, random_state)
    return {
        "sebelum": ringkas(algoritma, Xs_d, lab_d), "sesudah": ringkas(algoritma, Xs_g, lab_g),
        "Xs": Xs_g, "labels": lab_g, "baru": np.asarray(baru), "n_dasar": int((~baru).sum()),
        "n_tambahan": int(baru.sum()),
    }


# ===== UI =====
st.title("Clustering")
st.caption("Mengelompokkan data tanpa label menggunakan K-Means, Hierarchical, dan DBSCAN.")


@st.cache_data(show_spinner="Mengunduh dataset dari UCI...")
def load_uci_data(dataset_id):
    repo = fetch_ucirepo(id=dataset_id)
    return repo.data.features


tab_data, tab_hasil, tab_k, tab_banding, tab_tambah, tab_unggah = st.tabs(
    ["Data", "Hasil Clustering", "Penentuan K", "Bandingkan Algoritma", "Data Tambahan & Retraining", "Unggah Dataset"]
)

with tab_data:
    _next = st.session_state.pop("pilih_next_clustering", None)
    if _next:
        st.session_state["pilih_ds_clustering"] = _next
    opsi_dataset = list(UCI_CLUSTERING_DATASETS.keys()) + daftar_unggahan("clustering", st.session_state.get("data_ver", 0))
    if st.session_state.get("pilih_ds_clustering") not in opsi_dataset:
        st.session_state.pop("pilih_ds_clustering", None)
    pilihan_dataset = st.selectbox("Dataset", opsi_dataset, key="pilih_ds_clustering")

def _uci():
    return load_uci_data(UCI_CLUSTERING_DATASETS[pilihan_dataset]), []


ambil_uci = _uci if pilihan_dataset in UCI_CLUSTERING_DATASETS else None
try:
    semua, meta, peringatan = muat_dataset(
        pilihan_dataset, "clustering", st.session_state.get("data_ver", 0), ambil_uci
    )
except Exception as e:
    st.error(f"Gagal memuat dataset: {e}")
    st.stop()
if peringatan:
    with tab_data:
        st.warning(peringatan)

fitur_kolom = list(meta["kolom"])
X_raw = semua[fitur_kolom]
X_num, Xs = siapkan_fitur(X_raw)
sig = f"{pilihan_dataset}#{len(X_raw)}"


with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah baris", X_num.shape[0])
    c2.metric("Jumlah fitur", X_num.shape[1])
    c3.metric("Nilai kosong", int(X_raw.isnull().sum().sum()))
    st.dataframe(X_raw.head(20), use_container_width=True)
    st.write("Statistik deskriptif")
    st.dataframe(X_raw.describe().T, use_container_width=True)
    st.caption("Data distandarisasi sebelum clustering agar fitur berskala besar tidak mendominasi jarak.")

with tab_hasil:
    q1, q2, q3 = st.columns(3)
    algoritma = q1.selectbox("Algoritma", ALGORITMA)
    k = q2.slider("Jumlah cluster (K)", 2, 10, 3)
    random_state = int(q3.number_input("Random state", 0, 1000, 42, step=1))
    q4, q5, q6 = st.columns(3)
    metode_link = q4.selectbox("Linkage (Hierarchical)", ["ward", "complete", "average", "single"])
    eps = q5.slider("eps (DBSCAN)", 0.1, 5.0, 1.5, step=0.1)
    min_samples = q6.slider("min_samples (DBSCAN)", 2, 20, 5)
    jalankan_klik = st.button("Jalankan clustering", type="primary")

    if jalankan_klik:
        st.session_state["clu_hasil"] = {
            "dataset": sig,
            "algoritma": algoritma,
            "labels": jalankan(algoritma, Xs, k, metode_link, eps, min_samples, random_state),
        }

    tersimpan = st.session_state.get("clu_hasil")
    if not tersimpan or tersimpan["dataset"] != sig:
        st.info("Atur parameter di atas, lalu klik **Jalankan clustering**.")
    else:
        labels = tersimpan["labels"]
        algoritma = tersimpan["algoritma"]
        ring = ringkas(algoritma, Xs, labels)


        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Jumlah cluster", ring["Jumlah cluster"])
        m2.metric("Silhouette", "-" if ring["Silhouette"] is None else f"{ring['Silhouette']:.4f}")
        m3.metric("Davies-Bouldin", "-" if ring["Davies-Bouldin"] is None else f"{ring['Davies-Bouldin']:.4f}")
        m4.metric("Titik noise", ring["Titik noise"])
        st.caption("Silhouette makin mendekati 1 makin baik. Davies-Bouldin makin kecil makin baik.")

        if ring["Silhouette"] is None:
            st.warning("Hasil hanya memiliki satu cluster atau semua titik noise. "
                       "Untuk DBSCAN, naikkan eps atau turunkan min_samples.")

        pca = PCA(n_components=2).fit(Xs)
        P = pca.transform(Xs)
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, lab in enumerate(sorted(set(labels))):
            mask = labels == lab
            nama_lab = "Noise" if lab == -1 else f"Cluster {lab}"
            warna = "#888888" if lab == -1 else PALET[i % len(PALET)]
            ax.scatter(P[mask, 0], P[mask, 1], s=40, alpha=0.8, c=warna,
                       edgecolors="white", linewidth=0.4, label=nama_lab)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
        ax.set_title(f"{algoritma} (proyeksi PCA 2D)")
        ax.legend()
        ax.grid(alpha=0.2)
        st.pyplot(fig)
        plt.close(fig)

        st.write("Jumlah anggota per cluster")
        ukuran = pd.Series(labels).map(lambda v: "Noise" if v == -1 else f"Cluster {v}").value_counts()
        st.bar_chart(ukuran)

        st.write("Rata-rata fitur per cluster")
        profil = X_num.copy()
        profil["Cluster"] = labels
        st.dataframe(profil.groupby("Cluster").mean().round(3), use_container_width=True)

        if algoritma == "Hierarchical":
            st.write("Dendrogram")
            Z = linkage(Xs, method=metode_link)
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            dendrogram(Z, truncate_mode="lastp", p=30, ax=ax2)
            ax2.set_xlabel("Cluster (dipangkas 30 cabang terakhir)")
            ax2.set_ylabel("Jarak")
            st.pyplot(fig2)
            plt.close(fig2)

        if st.button("Simpan hasil ke MongoDB", key="simpan_clu"):
            try:
                simpan_eksperimen({
                    "jenis": "clustering", "dataset": pilihan_dataset,
                    "algoritma": algoritma,
                    "parameter": {"k": k, "linkage": metode_link, "eps": eps,
                                  "min_samples": min_samples},
                    "metrik": {
                        "jumlah_cluster": ring["Jumlah cluster"],
                        "noise": ring["Titik noise"],
                        "silhouette": None if ring["Silhouette"] is None else round(ring["Silhouette"], 4),
                        "davies_bouldin": None if ring["Davies-Bouldin"] is None else round(ring["Davies-Bouldin"], 4),
                    },
                })
                st.success("Hasil tersimpan di MongoDB.")
            except Exception as e:
                st.error(f"Gagal menyimpan: {e}")

with tab_k:
    st.write("Elbow Method dan Silhouette untuk K-Means membantu memilih jumlah cluster.")
    k_min, k_max = st.slider("Rentang K", 2, 12, (2, 8))
    ks, inertia, sil = elbow(Xs, k_min, k_max, random_state)
    col_a, col_b = st.columns(2)
    with col_a:
        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        ax3.plot(ks, inertia, marker="o", color="#4DA3FF")
        ax3.set_xlabel("K")
        ax3.set_ylabel("Inertia")
        ax3.set_title("Elbow Method")
        ax3.grid(alpha=0.3)
        st.pyplot(fig3)
        plt.close(fig3)
    with col_b:
        fig4, ax4 = plt.subplots(figsize=(5, 3.5))
        ax4.plot(ks, sil, marker="o", color="#2ED3A0")
        ax4.set_xlabel("K")
        ax4.set_ylabel("Silhouette")
        ax4.set_title("Silhouette per K")
        ax4.grid(alpha=0.3)
        st.pyplot(fig4)
        plt.close(fig4)
    st.info(f"K dengan silhouette tertinggi: {ks[int(np.argmax(sil))]}. "
            "Cocokkan dengan titik siku pada grafik Elbow sebelum memutuskan.")

with tab_banding:
    st.write("Jalankan ketiga algoritma dengan parameter di tab Hasil Clustering, lalu bandingkan.")
    if st.button("Bandingkan algoritma"):
        baris = [ringkas(n, Xs, jalankan(n, Xs, k, metode_link, eps, min_samples, random_state))
                 for n in ALGORITMA]
        st.session_state["clu_banding"] = {"dataset": sig, "tabel": pd.DataFrame(baris)}
    b = st.session_state.get("clu_banding")
    if b and b["dataset"] == sig:
        tabel = b["tabel"]
        st.dataframe(tabel.round(4), use_container_width=True)
        valid = tabel.dropna(subset=["Silhouette"])
        if valid.empty:
            st.warning("Tidak ada algoritma yang menghasilkan minimal dua cluster.")
        else:
            terbaik = valid.sort_values("Silhouette", ascending=False).iloc[0]
            st.info(f"Silhouette tertinggi: {terbaik['Algoritma']} ({terbaik['Silhouette']:.3f}). "
                    "DBSCAN tidak memerlukan K dan dapat menandai outlier sebagai noise.")


with tab_tambah:
    st.write("Data di sini khusus untuk retraining. Data ini **tidak digabung** ke dataset utama, "
             "jadi tab Data dan Latih Model tidak berubah.")
    tam = muat_tambahan(pilihan_dataset, "clustering", st.session_state.get("tam_ver", 0),
                        tuple(meta["kolom"]), tuple(meta["numerik"]))
    n_tambah = len(tam)
    render_form(pilihan_dataset, "clustering", semua, tam, meta, fitur_kolom, "clu")

    st.divider()
    st.subheader("Clustering ulang")
    st.write(f"Dataset utama: {len(X_raw)} baris | Data tambahan: {n_tambah} baris")
    st.caption(f"Memakai algoritma dan parameter dari tab Hasil Clustering: {algoritma}.")
    sig_rt = f"{sig}+{n_tambah}"

    if st.button("Kelompokkan ulang dan bandingkan", type="primary", key="clu_rt_btn"):
        if n_tambah == 0:
            st.warning("Belum ada data tambahan. Isi form di atas dulu.")
        else:
            X_cat = pd.concat([X_raw, tam[fitur_kolom]], ignore_index=True)
            baru = np.array([False] * len(X_raw) + [True] * n_tambah)
            with st.spinner("Mengelompokkan sebelum dan sesudah penambahan data..."):
                hasil = retrain_clustering(
                    algoritma, X_cat, baru, k, metode_link, eps, min_samples, random_state
                )
            hasil.update(sig=sig_rt, algoritma=algoritma)
            st.session_state["clu_rt"] = hasil
            try:
                catat_eksperimen({
                    "jenis": "retraining_clustering", "dataset": pilihan_dataset, "algoritma": algoritma,
                    "n_dasar": hasil["n_dasar"], "n_tambahan": hasil["n_tambahan"],
                    "sebelum": {m: (None if v is None else round(v, 4)) for m, v in hasil["sebelum"].items() if m != "Algoritma"},
                    "sesudah": {m: (None if v is None else round(v, 4)) for m, v in hasil["sesudah"].items() if m != "Algoritma"},
                })
            except Exception as e:
                st.warning(f"Hasil tidak tersimpan ke MongoDB: {e}")

    r = st.session_state.get("clu_rt")
    if r and r["sig"] == sig_rt:
        st.success(f"{r['algoritma']}: {r['n_dasar']} baris menjadi {r['n_dasar'] + r['n_tambahan']} baris.")
        tampilkan_perbandingan(
            r["sebelum"], r["sesudah"],
            ["Jumlah cluster", "Silhouette", "Davies-Bouldin", "Titik noise"],
            lebih_kecil=("Davies-Bouldin",), netral=("Jumlah cluster", "Titik noise"),
        )
        P = PCA(n_components=2).fit_transform(r["Xs"])
        lab = np.asarray(r["labels"])
        baru_p = r["baru"]
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        for i, lb in enumerate(sorted(set(lab))):
            warna = "#888888" if lb == -1 else PALET[i % len(PALET)]
            m = lab == lb
            nama_lb = "Noise" if lb == -1 else f"Cluster {lb}"
            ax5.scatter(P[m & ~baru_p, 0], P[m & ~baru_p, 1], s=35, c=warna, alpha=0.8, label=nama_lb)
            ax5.scatter(P[m & baru_p, 0], P[m & baru_p, 1], s=90, c=warna, marker="^",
                        edgecolors="black", linewidth=1)
        ax5.set_title("Hasil sesudah penambahan data (segitiga = data tambahan)")
        ax5.legend()
        ax5.grid(alpha=0.2)
        st.pyplot(fig5)
        plt.close(fig5)

with tab_unggah:
    render_unggah_dataset("clustering", "cluu", None, opsi_dataset, pilihan_dataset, list(UCI_CLUSTERING_DATASETS.keys())[0])