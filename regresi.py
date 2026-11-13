import streamlit as st
from data_tambahan import (
    muat_dataset, muat_tambahan, daftar_unggahan, render_form, render_unggah_dataset,
    catat_eksperimen, tampilkan_perbandingan,
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils import terapkan_tema
terapkan_tema("regresi")


UCI_REGRESSION_DATASETS = {
    "Real Estate Valuation": 477,
    "Concrete Compressive Strength": 165,
    "Energy Efficiency": 242,
    
}

MODEL_NAMES = [
    "Linear Regression", "Ridge", "Lasso",
    "Decision Tree", "Random Forest", "Gradient Boosting",
]


def make_model(nama, random_state):
    model = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1, max_iter=10000),
        "Decision Tree": DecisionTreeRegressor(random_state=random_state),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state),
        "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
    }[nama]
    # Imputasi dilakukan di dalam pipeline agar tidak bocor dari data uji
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def preprocess_fitur(X):
    X = X.copy()
    # Kolom tanggal tidak bermakna bila di-encode sebagai angka urut
    tanggal = [c for c in X.columns if str(c).strip().lower() == "date"]
    X = X.drop(columns=tanggal)
    # One-hot encoding untuk kolom teks/kategori
    X = pd.get_dummies(X, drop_first=True)
    return X.astype(float)


def evaluasi(nama, X_train, X_test, y_train, y_test, random_state):
    pipe = make_model(nama, random_state)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    return pipe, pred, {
        "Model": nama,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_test, pred)),
        "R2": float(r2_score(y_test, pred)),
    }


def pentingnya_fitur(pipe, kolom):
    m = pipe.named_steps["model"]
    if hasattr(m, "feature_importances_"):
        nilai = m.feature_importances_
    elif hasattr(m, "coef_"):
        nilai = np.abs(np.ravel(m.coef_))
    else:
        return None
    return pd.Series(nilai, index=kolom).sort_values(ascending=False)



def retrain_regresi(nama, X_raw, y, baru, test_size, random_state):
    """Latih dari data UCI (sebelum) dan UCI + tambahan (sesudah); data uji diambil dari data UCI dan sama untuk keduanya."""
    X_all = preprocess_fitur(X_raw).reset_index(drop=True)
    y_all = y.reset_index(drop=True)
    idx_dasar = np.where(~baru)[0]
    idx_baru = np.where(baru)[0]
    tr, te = train_test_split(idx_dasar, test_size=test_size, random_state=random_state)
    tr2 = np.concatenate([tr, idx_baru])
    _, _, sebelum = evaluasi(nama, X_all.iloc[tr], X_all.iloc[te], y_all.iloc[tr], y_all.iloc[te], random_state)
    _, _, sesudah = evaluasi(nama, X_all.iloc[tr2], X_all.iloc[te], y_all.iloc[tr2], y_all.iloc[te], random_state)
    return {
        "sebelum": sebelum, "sesudah": sesudah, "n_latih_sebelum": len(tr),
        "n_latih_sesudah": len(tr2), "n_uji": len(te), "n_tambahan": len(idx_baru),
    }


# ===== UI =====
st.title("Regresi")
st.caption("Prediksi nilai numerik menggunakan dataset UCI Machine Learning Repository.")


@st.cache_data(show_spinner="Mengunduh dataset dari UCI...")
def load_uci_data(dataset_id):
    repo = fetch_ucirepo(id=dataset_id)
    return repo.data.features, repo.data.targets


tab_data, tab_latih, tab_banding, tab_tambah, tab_unggah = st.tabs(
    ["Data", "Latih Model", "Bandingkan Semua", "Data Tambahan & Retraining", "Unggah Dataset"]
)

with tab_data:
    _next = st.session_state.pop("pilih_next_regresi", None)
    if _next:
        st.session_state["pilih_ds_regresi"] = _next
    opsi_dataset = list(UCI_REGRESSION_DATASETS.keys()) + daftar_unggahan("regresi", st.session_state.get("data_ver", 0))
    if st.session_state.get("pilih_ds_regresi") not in opsi_dataset:
        st.session_state.pop("pilih_ds_regresi", None)
    pilihan_dataset = st.selectbox("Dataset", opsi_dataset, key="pilih_ds_regresi")

def _uci():
    X_uci, y_uci = load_uci_data(UCI_REGRESSION_DATASETS[pilihan_dataset])
    return pd.concat([X_uci, y_uci], axis=1), list(y_uci.columns)


ambil_uci = _uci if pilihan_dataset in UCI_REGRESSION_DATASETS else None
try:
    semua, meta, peringatan = muat_dataset(
        pilihan_dataset, "regresi", st.session_state.get("data_ver", 0), ambil_uci
    )
except Exception as e:
    st.error(f"Gagal memuat dataset: {e}")
    st.stop()
if peringatan:
    with tab_data:
        st.warning(peringatan)

# Beberapa dataset (mis. Energy Efficiency) punya lebih dari satu target
target_semua = meta["target"]
target = target_semua[0]
if len(target_semua) > 1:
    with tab_data:
        target = st.selectbox("Kolom target", target_semua)
fitur_kolom = [c for c in meta["kolom"] if c not in target_semua]

semua[target] = pd.to_numeric(semua[target], errors="coerce")
semua = semua[semua[target].notna()].reset_index(drop=True)
X_raw = semua[fitur_kolom]
y = semua[target].astype(float)
X = preprocess_fitur(X_raw)
kunci_data = pilihan_dataset if len(target_semua) == 1 else f"{pilihan_dataset} [{target}]"
sig = f"{kunci_data}#{len(X)}"


with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah baris", X.shape[0])
    c2.metric("Jumlah fitur", X.shape[1])
    c3.metric("Nilai kosong", int(X_raw.isnull().sum().sum()))
    st.write(f"Target: **{target}**")
    st.dataframe(pd.concat([X_raw, y], axis=1).head(20), use_container_width=True)
    st.write("Statistik deskriptif")
    st.dataframe(X_raw.describe().T, use_container_width=True)

with tab_latih:
    p1, p2 = st.columns(2)
    test_size = p1.slider("Test size", 0.1, 0.5, 0.2, step=0.05)
    random_state = p2.number_input("Random state", 0, 1000, 42, step=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=int(random_state)
)

with tab_latih:
    pilihan_model = st.selectbox("Model", MODEL_NAMES)
    if st.button("Latih model", type="primary"):
        with st.spinner("Melatih model..."):
            pipe, pred, hasil = evaluasi(
                pilihan_model, X_train, X_test, y_train, y_test, int(random_state)
            )
        st.session_state["reg_latih"] = {
            "dataset": sig, "pipe": pipe, "pred": pred, "hasil": hasil,
            "y_test": y_test.values, "kolom": list(X.columns),
        }

    r = st.session_state.get("reg_latih")
    if r and r["dataset"] == sig:
        h = r["hasil"]
        st.success(f"Hasil {h['Model']} pada {pilihan_dataset}")
        m1, m2, m3 = st.columns(3)
        m1.metric("RMSE", f"{h['RMSE']:.3f}")
        m2.metric("MAE", f"{h['MAE']:.3f}")
        m3.metric("R²", f"{h['R2']:.4f}")

        y_t, y_p = r["y_test"], r["pred"]
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        ax[0].scatter(y_t, y_p, alpha=0.6, color="#4DA3FF")
        ax[0].plot([y_t.min(), y_t.max()], [y_t.min(), y_t.max()], "r--", lw=2)
        ax[0].set_xlabel("Nilai aktual")
        ax[0].set_ylabel("Nilai prediksi")
        ax[0].set_title("Aktual vs Prediksi")
        sns.scatterplot(x=y_p, y=y_t - y_p, ax=ax[1], color="#FF6B6B", alpha=0.6)
        ax[1].axhline(0, color="black", linestyle="--")
        ax[1].set_xlabel("Nilai prediksi")
        ax[1].set_ylabel("Residual")
        ax[1].set_title("Residual Plot")
        st.pyplot(fig)
        plt.close(fig)

        penting = pentingnya_fitur(r["pipe"], r["kolom"])
        if penting is not None:
            st.write("Fitur paling berpengaruh")
            st.bar_chart(penting.head(10))

with tab_banding:
    st.write("Latih semua model dengan pengaturan yang sama, lalu bandingkan.")
    if st.button("Bandingkan semua model"):
        baris = []
        bar = st.progress(0.0)
        for i, nama in enumerate(MODEL_NAMES):
            _, _, h = evaluasi(nama, X_train, X_test, y_train, y_test, int(random_state))
            baris.append(h)
            bar.progress((i + 1) / len(MODEL_NAMES))
        st.session_state["reg_banding"] = {
            "dataset": sig,
            "tabel": pd.DataFrame(baris).sort_values("RMSE").reset_index(drop=True),
        }

    b = st.session_state.get("reg_banding")
    if b and b["dataset"] == sig:
        tabel = b["tabel"]
        st.dataframe(tabel.round(4), use_container_width=True)
        st.bar_chart(tabel.set_index("Model")["RMSE"])
        terbaik = tabel.iloc[0]
        st.info(
            f"Model terbaik: {terbaik['Model']} dengan RMSE {terbaik['RMSE']:.3f} "
            f"dan R² {terbaik['R2']:.3f}. RMSE lebih kecil berarti kesalahan prediksi lebih kecil."
        )


with tab_tambah:
    st.write("Data di sini khusus untuk retraining. Data ini **tidak digabung** ke dataset utama, "
             "jadi tab Data dan Latih Model tidak berubah.")
    tam = muat_tambahan(pilihan_dataset, "regresi", st.session_state.get("tam_ver", 0),
                        tuple(meta["kolom"]), tuple(meta["numerik"]))
    tam = tam[pd.to_numeric(tam[target], errors="coerce").notna()].reset_index(drop=True)
    n_tambah = len(tam)
    render_form(pilihan_dataset, "regresi", semua, tam, meta, fitur_kolom, "reg", target=(target, "angka"))

    st.divider()
    st.subheader("Retraining")
    st.write(f"Dataset utama: {len(X_raw)} baris | Data tambahan: {n_tambah} baris")

    model_rt = st.selectbox("Model untuk retraining", MODEL_NAMES, key="reg_rt_model")
    sig_rt = f"{sig}+{n_tambah}"
    if st.button("Latih ulang dan bandingkan", type="primary", key="reg_rt_btn"):
        if n_tambah == 0:
            st.warning("Belum ada data tambahan. Isi form di atas dulu.")
        else:
            X_cat = pd.concat([X_raw, tam[fitur_kolom]], ignore_index=True)
            y_cat = pd.concat([y.reset_index(drop=True), pd.to_numeric(tam[target]).astype(float)], ignore_index=True)
            baru = np.array([False] * len(X_raw) + [True] * n_tambah)
            with st.spinner("Melatih sebelum dan sesudah penambahan data..."):
                hasil = retrain_regresi(model_rt, X_cat, y_cat, baru, test_size, int(random_state))
            hasil.update(sig=sig_rt, model=model_rt)
            st.session_state["reg_rt"] = hasil
            try:
                catat_eksperimen({
                    "jenis": "retraining_regresi", "dataset": kunci_data, "model": model_rt,
                    "n_dasar": len(X_raw), "n_tambahan": n_tambah,
                    "sebelum": {k: round(v, 4) for k, v in hasil["sebelum"].items() if k != "Model"},
                    "sesudah": {k: round(v, 4) for k, v in hasil["sesudah"].items() if k != "Model"},
                })
            except Exception as e:
                st.warning(f"Hasil tidak tersimpan ke MongoDB: {e}")

    r = st.session_state.get("reg_rt")
    if r and r["sig"] == sig_rt:
        st.success(
            f"{r['model']}: data latih {r['n_latih_sebelum']} baris menjadi {r['n_latih_sesudah']} baris "
            f"(data uji tetap {r['n_uji']} baris)."
        )
        tampilkan_perbandingan(r["sebelum"], r["sesudah"], ["RMSE", "MAE", "R2"], lebih_kecil=("RMSE", "MAE"))
        tabel = pd.DataFrame({
            "Sebelum": {m: r["sebelum"][m] for m in ("RMSE", "MAE", "R2")},
            "Sesudah": {m: r["sesudah"][m] for m in ("RMSE", "MAE", "R2")},
        })
        st.dataframe(tabel.round(4), use_container_width=True)
        st.caption("Data uji sama untuk kedua model agar perbandingannya adil. Panah hijau berarti membaik.")


with tab_unggah:
    render_unggah_dataset("regresi", "regu", "angka", opsi_dataset, pilihan_dataset, list(UCI_REGRESSION_DATASETS.keys())[0])