import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timezone
from utils import terapkan_tema



try:
    import tensorflow as tf
    keras = tf.keras
    layers = tf.keras.layers
except ImportError:
    tf = None

terapkan_tema("CNN")

def siapkan_gambar(X):
    return X.reshape(-1, 28, 28, 1).astype("float32") / 255.0


def bangun_model(filter1, filter2, unit_dense, dropout):
    return keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(filter1, kernel_size=5, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=2),
        layers.Conv2D(filter2, kernel_size=5, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=2),
        layers.Dropout(dropout),
        layers.Flatten(),
        layers.Dense(unit_dense, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(10, activation="softmax"),
    ])


def buat_callback(bar, teks, total_epoch):
    class ProgresStreamlit(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            bar.progress((epoch + 1) / total_epoch)
            teks.write(
                f"Epoch {epoch + 1}/{total_epoch} | "
                f"loss {logs.get('loss', 0):.4f} | akurasi {logs.get('accuracy', 0):.4f} | "
                f"val_akurasi {logs.get('val_accuracy', 0):.4f}"
            )
    return ProgresStreamlit()


def simpan_eksperimen(dokumen):
    from db import get_db
    dokumen["waktu"] = datetime.now(timezone.utc)
    get_db()["eksperimen"].insert_one(dokumen)


# ===== UI =====
terapkan_tema("deep_learning")
st.title("Deep Learning: CNN pada MNIST")
st.caption("Jaringan saraf konvolusi (CNN) untuk mengenali angka tulisan tangan 0-9.")

if tf is None:
    st.error("TensorFlow belum terpasang. Jalankan: pip install tensorflow")
    st.stop()


@st.cache_data(show_spinner="Mengunduh dataset MNIST...")
def load_mnist():
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    return X_train, y_train, X_test, y_test


try:
    X_train, y_train, X_test, y_test = load_mnist()
except Exception as e:
    st.error(f"Gagal mengunduh MNIST: {e}")
    st.stop()

tab_info, tab_latih, tab_eval = st.tabs(["Informasi Data", "Pelatihan Model", "Evaluasi & Prediksi"])

with tab_info:
    c1, c2, c3 = st.columns(3)
    c1.metric("Data latih", f"{X_train.shape[0]:,}")
    c2.metric("Data uji", f"{X_test.shape[0]:,}")
    c3.metric("Ukuran gambar", "28 x 28 piksel")
    st.write("Sepuluh contoh gambar (grayscale)")
    fig, axes = plt.subplots(1, 10, figsize=(12, 1.8))
    for i in range(10):
        axes[i].imshow(X_train[i], cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(str(y_train[i]))
    st.pyplot(fig)
    plt.close(fig)
    st.write("Distribusi kelas pada data latih")
    st.bar_chart(pd.Series(y_train).value_counts().sort_index())

with tab_latih:
    c1, c2, c3 = st.columns(3)
    jumlah = c1.select_slider("Jumlah data latih", options=[5000, 10000, 20000, 60000], value=10000)
    epochs = c2.slider("Epochs", 1, 20, 5)
    batch_size = c3.select_slider("Batch size", options=[32, 64, 128, 256], value=128)
    c4, c5 = st.columns(2)
    dropout = c4.slider("Dropout", 0.0, 0.6, 0.5, step=0.1)
    unit_dense = c5.select_slider("Unit lapisan Dense", options=[64, 128, 256], value=128)
    st.caption("Mulai dengan 10.000 data dan 5 epoch agar cepat. Data 60.000 butuh waktu lebih lama di server.")

    if st.button("Mulai", type="primary"):
        model = bangun_model(16, 32, unit_dense, dropout)
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        bar = st.progress(0.0)
        teks = st.empty()
        with st.spinner("Melatih model..."):
            riwayat = model.fit(
                siapkan_gambar(X_train[:jumlah]), y_train[:jumlah],
                validation_split=0.2, epochs=epochs, batch_size=batch_size,
                verbose=0, callbacks=[buat_callback(bar, teks, epochs)],
            )
        st.session_state["dl_model"] = model
        st.session_state["dl_info"] = {
            "riwayat": riwayat.history, "jumlah": jumlah, "epochs": epochs,
            "batch_size": batch_size, "dropout": dropout, "unit_dense": unit_dense,
        }
        st.session_state.pop("dl_tes", None)
        st.success("Pelatihan selesai. Buka tab Evaluasi & Prediksi untuk melihat hasilnya.")

    info = st.session_state.get("dl_info")
    if info:
        h = info["riwayat"]
        rentang = range(1, len(h["loss"]) + 1)
        g1, g2 = st.columns(2)
        with g1:
            fig_l, ax_l = plt.subplots()
            ax_l.plot(rentang, h["loss"], "o-", label="Training")
            ax_l.plot(rentang, h["val_loss"], "o--", label="Validasi")
            ax_l.set_title("Loss")
            ax_l.set_xlabel("Epoch")
            ax_l.legend()
            st.pyplot(fig_l)
            plt.close(fig_l)
        with g2:
            fig_a, ax_a = plt.subplots()
            ax_a.plot(rentang, h["accuracy"], "o-", label="Training")
            ax_a.plot(rentang, h["val_accuracy"], "o--", label="Validasi")
            ax_a.set_title("Akurasi")
            ax_a.set_xlabel("Epoch")
            ax_a.legend()
            st.pyplot(fig_a)
            plt.close(fig_a)

with tab_eval:
    model = st.session_state.get("dl_model")
    if model is None:
        st.info("Model belum dilatih. Latih dulu di tab Pelatihan Model.")
    else:
        if "dl_tes" not in st.session_state:
            with st.spinner("Mengevaluasi pada data uji..."):
                Xt = siapkan_gambar(X_test)
                probabilitas = model.predict(Xt, verbose=0)
                prediksi = np.argmax(probabilitas, axis=1)
                st.session_state["dl_tes"] = {
                    "pred": prediksi,
                    "akurasi": float((prediksi == y_test).mean()),
                }
        tes = st.session_state["dl_tes"]
        st.metric("Akurasi data uji", f"{tes['akurasi'] * 100:.2f}%")

        if st.checkbox("Tampilkan confusion matrix"):
            cm = pd.crosstab(y_test, tes["pred"], rownames=["Aktual"], colnames=["Prediksi"])
            fig_c, ax_c = plt.subplots(figsize=(7, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_c)
            st.pyplot(fig_c)
            plt.close(fig_c)

        st.divider()
        st.write("Coba prediksi satu gambar")
        idx = st.slider("Indeks gambar uji", 0, len(X_test) - 1, 7)
        p = model.predict(siapkan_gambar(X_test[idx:idx + 1]), verbose=0)[0]
        kelas = int(np.argmax(p))
        k1, k2 = st.columns(2)
        with k1:
            st.write(f"Label asli: **{y_test[idx]}**")
            fig_i, ax_i = plt.subplots(figsize=(3, 3))
            ax_i.imshow(X_test[idx], cmap="gray")
            ax_i.axis("off")
            st.pyplot(fig_i)
            plt.close(fig_i)
        with k2:
            benar = "benar" if kelas == int(y_test[idx]) else "salah"
            st.write(f"Prediksi model: **{kelas}** ({p[kelas] * 100:.2f}%), {benar}")
            fig_p, ax_p = plt.subplots(figsize=(5, 3))
            ax_p.bar(np.arange(10), p, color="#4DA3FF")
            ax_p.set_xticks(np.arange(10))
            ax_p.set_title("Probabilitas tiap angka")
            st.pyplot(fig_p)
            plt.close(fig_p)

        st.divider()
        if st.button("Simpan hasil ke MongoDB"):
            info = st.session_state["dl_info"]
            try:
                simpan_eksperimen({
                    "jenis": "deep_learning", "dataset": "MNIST", "model": "CNN",
                    "parameter": {k: info[k] for k in ("jumlah", "epochs", "batch_size", "dropout", "unit_dense")},
                    "metrik": {
                        "akurasi_uji": round(tes["akurasi"], 4),
                        "akurasi_validasi": round(float(info["riwayat"]["val_accuracy"][-1]), 4),
                    },
                })
                st.success("Hasil tersimpan di MongoDB.")
            except Exception as e:
                st.error(f"Gagal menyimpan: {e}")