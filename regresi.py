import streamlit as st
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
    "Seoul Bike Sharing Demand": 560,
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


# ===== UI =====
st.title("Regresi")
st.caption("Prediksi nilai numerik menggunakan dataset UCI Machine Learning Repository.")


@st.cache_data(show_spinner="Mengunduh dataset dari UCI...")
def load_uci_data(dataset_id):
    repo = fetch_ucirepo(id=dataset_id)
    return repo.data.features, repo.data.targets


with st.sidebar:
    st.header("Pengaturan")
    pilihan_dataset = st.selectbox("Dataset UCI", list(UCI_REGRESSION_DATASETS.keys()))
    test_size = st.slider("Test size", 0.1, 0.5, 0.2, step=0.05)
    random_state = st.number_input("Random state", 0, 1000, 42, step=1)

try:
    X_raw, y_raw = load_uci_data(UCI_REGRESSION_DATASETS[pilihan_dataset])
except Exception as e:
    st.error(f"Gagal mengunduh dataset: {e}")
    st.stop()

# Beberapa dataset (mis. Energy Efficiency) punya lebih dari satu target
target = y_raw.columns[0]
if y_raw.shape[1] > 1:
    target = st.sidebar.selectbox("Kolom target", list(y_raw.columns))

y = y_raw[target].astype(float)
X = preprocess_fitur(X_raw)

tab_data, tab_latih, tab_banding = st.tabs(["Data", "Latih Model", "Bandingkan Semua"])

with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah baris", X.shape[0])
    c2.metric("Jumlah fitur", X.shape[1])
    c3.metric("Nilai kosong", int(X_raw.isnull().sum().sum()))
    st.write(f"Target: **{target}**")
    st.dataframe(pd.concat([X_raw, y], axis=1).head(20), use_container_width=True)
    st.write("Statistik deskriptif")
    st.dataframe(X_raw.describe().T, use_container_width=True)

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
            "dataset": pilihan_dataset, "pipe": pipe, "pred": pred, "hasil": hasil,
            "y_test": y_test.values, "kolom": list(X.columns),
        }

    r = st.session_state.get("reg_latih")
    if r and r["dataset"] == pilihan_dataset:
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
            "dataset": pilihan_dataset,
            "tabel": pd.DataFrame(baris).sort_values("RMSE").reset_index(drop=True),
        }

    b = st.session_state.get("reg_banding")
    if b and b["dataset"] == pilihan_dataset:
        tabel = b["tabel"]
        st.dataframe(tabel.round(4), use_container_width=True)
        st.bar_chart(tabel.set_index("Model")["RMSE"])
        terbaik = tabel.iloc[0]
        st.info(
            f"Model terbaik: {terbaik['Model']} dengan RMSE {terbaik['RMSE']:.3f} "
            f"dan R² {terbaik['R2']:.3f}. RMSE lebih kecil berarti kesalahan prediksi lebih kecil."
        )