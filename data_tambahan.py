import json
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timezone
from db import get_db

SUMBER_UTAMA = ["uci", "unggahan"]        # baris yang membentuk dataset di dropdown
SUMBER_TAMBAHAN = ["tambahan", "pengguna"]  # baris khusus retraining (tidak digabung ke dataset utama)
BATCH = 2000
MIN_BARIS = 20
MAKS_BARIS_DATASET = 5000
MAKS_KOLOM = 50
MAKS_DATASET_UNGGAHAN = 10
MAKS_TAMBAHAN = 5000


def _versi():
    return st.session_state.get("data_ver", 0)


def _naik_versi(nama="data_ver"):
    st.session_state[nama] = st.session_state.get(nama, 0) + 1


def _py(v):
    if v is None:
        return None
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def _numerik(s):
    return bool(pd.api.types.is_numeric_dtype(s))


# ---------------------------------------------------------------- dataset utama
def salin_ke_mongo(db, kunci, jenis, df, target, sumber, nama_file=None):
    """Simpan dataset utama (UCI atau unggahan) ke MongoDB."""
    kolom = [str(c) for c in df.columns]
    numerik = [_numerik(df[c]) for c in df.columns]
    nilai = json.loads(df.to_json(orient="values"))
    db["dataset_rows"].delete_many({"dataset": kunci, "jenis": jenis})
    db["datasets"].delete_many({"dataset": kunci, "jenis": jenis})
    for i in range(0, len(nilai), BATCH):
        db["dataset_rows"].insert_many([
            {"dataset": kunci, "jenis": jenis, "sumber": sumber, "i": i + j, "nilai": v}
            for j, v in enumerate(nilai[i:i + BATCH])
        ])
    db["datasets"].insert_one({
        "dataset": kunci, "jenis": jenis, "sumber": sumber, "kolom": kolom,
        "numerik": numerik, "target": [str(t) for t in target], "nama_file": nama_file,
        "jumlah_baris": len(nilai), "jumlah_tambahan": 0, "waktu": datetime.now(timezone.utc),
    })


@st.cache_data(ttl=60)
def daftar_unggahan(jenis, versi):
    try:
        return sorted(d["dataset"] for d in get_db()["datasets"].find(
            {"jenis": jenis, "sumber": "unggahan"}, {"dataset": 1}))
    except Exception:
        return []


@st.cache_data(ttl=120, show_spinner="Memuat dataset dari MongoDB...")
def muat_dataset(kunci, jenis, versi, _ambil_uci):
    """
    Mengembalikan (df, meta, peringatan) untuk dataset utama.
    Dataset UCI otomatis disalin ke MongoDB saat pertama dibuka. Dataset unggahan
    sudah ada di MongoDB. Jika MongoDB tidak terjangkau, dataset UCI dipakai langsung.
    """
    try:
        db = get_db()
        filt = {"dataset": kunci, "jenis": jenis, "sumber": {"$in": SUMBER_UTAMA}}
        meta = db["datasets"].find_one(filt)
        if meta is None:
            if _ambil_uci is None:
                raise LookupError(f"Dataset '{kunci}' tidak ditemukan di MongoDB.")
            df, target = _ambil_uci()
            salin_ke_mongo(db, kunci, jenis, df, target, "uci")
            meta = db["datasets"].find_one(filt)
        baris = list(db["dataset_rows"].find(
            {"dataset": kunci, "jenis": jenis, "sumber": {"$in": SUMBER_UTAMA}}, {"nilai": 1}
        ).sort("i", 1))
        kolom = meta["kolom"]
        df = pd.DataFrame([b["nilai"] for b in baris], columns=kolom)
        for c, n in zip(kolom, meta["numerik"]):
            if n:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df, {"kolom": kolom, "numerik": list(meta["numerik"]), "target": list(meta["target"])}, ""
    except Exception as e:
        if _ambil_uci is None:
            raise
        df, target = _ambil_uci()
        df.columns = [str(c) for c in df.columns]
        info = {"kolom": list(df.columns), "numerik": [_numerik(df[c]) for c in df.columns],
                "target": [str(t) for t in target]}
        return df, info, f"MongoDB tidak terjangkau, memakai data UCI langsung ({e})"


# ---------------------------------------------------------------- data retraining (terpisah)
@st.cache_data(ttl=120)
def muat_tambahan(kunci, jenis, versi, kolom, numerik):
    """Baris tambahan khusus retraining. Tidak pernah digabung ke dataset utama."""
    try:
        baris = list(get_db()["dataset_rows"].find(
            {"dataset": kunci, "jenis": jenis, "sumber": {"$in": SUMBER_TAMBAHAN}}, {"nilai": 1}
        ).sort("waktu", 1))
        df = pd.DataFrame([b["nilai"] for b in baris], columns=list(kolom))
        for c, n in zip(kolom, numerik):
            if n:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame(columns=list(kolom))


def _widget(kol, s, is_num, key, wadah, label=None):
    label = label or kol
    s = s.dropna()
    if is_num:
        if s.empty:
            return wadah.number_input(label, value=0.0, key=key)
        lo, hi, med = float(s.min()), float(s.max()), float(s.median())
        rentang = f"Rentang data: {lo:g} sampai {hi:g}"
        if bool((s == s.round()).all()):
            return wadah.number_input(label, value=int(round(med)), step=1, help=rentang, key=key)
        langkah = float(f"{max((hi - lo) / 100, 1e-6):.1g}")
        return wadah.number_input(label, value=med, step=langkah, format="%.4f", help=rentang, key=key)
    u = s.astype(str).value_counts()
    if len(u) <= 50:
        return wadah.selectbox(label, list(u.index), key=key)
    return wadah.text_input(label, value=str(u.index[0]), key=key)


def render_form(kunci, jenis, semua, tambahan, meta, kolom_fitur, kp, target=None):
    """
    Form angka untuk menambah satu baris data retraining.
    semua    : dataset utama (dipakai untuk isian awal dan rentang)
    tambahan : data retraining yang sudah tersimpan
    target   : None, (nama, "angka") atau (nama, "kelas")
    """
    numerik = dict(zip(meta["kolom"], meta["numerik"]))
    pesan = st.session_state.pop(f"{kp}_pesan", None)
    if pesan:
        st.success(pesan)

    c1, c2 = st.columns(2)
    c1.metric("Data dataset utama", len(semua))
    c2.metric("Data tambahan retraining", len(tambahan))

    isi = {}
    with st.form(f"{kp}_form_{kunci}", clear_on_submit=False):
        st.write("Isi nilai tiap kolom, lalu klik **Tambah data**. Isian awal adalah nilai tengah dataset.")
        kol = st.columns(3)
        for i, c in enumerate(kolom_fitur):
            isi[c] = _widget(c, semua[c], numerik.get(c, True), f"{kp}_{kunci}_{c}", kol[i % 3])
        if target:
            nama, mode = target
            st.markdown(f"**Target: {nama}**")
            if mode == "angka":
                isi[nama] = _widget(nama, semua[nama], True, f"{kp}_{kunci}_{nama}", st, f"{nama} (nilai sebenarnya)")
            else:
                opsi = sorted(semua[nama].dropna().unique().tolist(), key=str)
                isi[nama] = st.selectbox(f"{nama} (kelas)", opsi, key=f"{kp}_{kunci}_{nama}")
        kirim = st.form_submit_button("Tambah data", type="primary")

    if kirim:
        try:
            if len(tambahan) >= MAKS_TAMBAHAN:
                st.error(f"Data tambahan dibatasi {MAKS_TAMBAHAN} baris per dataset.")
            else:
                db = get_db()
                db["dataset_rows"].insert_one({
                    "dataset": kunci, "jenis": jenis, "sumber": "tambahan",
                    "nilai": [_py(isi.get(c)) for c in meta["kolom"]],
                    "waktu": datetime.now(timezone.utc),
                })
                db["datasets"].update_one({"dataset": kunci, "jenis": jenis}, {"$inc": {"jumlah_tambahan": 1}})
                _naik_versi("tam_ver")
                st.session_state[f"{kp}_pesan"] = "Data ditambahkan untuk retraining. Dataset utama tidak berubah."
                st.rerun()
        except Exception as e:
            st.error(f"Gagal menyimpan: {e}")

    if len(tambahan):
        with st.expander("Lihat dan kelola data tambahan"):
            st.dataframe(tambahan.tail(20), use_container_width=True)
            yakin = st.checkbox("Saya yakin ingin menghapus semua data tambahan", key=f"{kp}_yakin")
            if st.button("Hapus data tambahan", disabled=not yakin, key=f"{kp}_hapus"):
                try:
                    db = get_db()
                    db["dataset_rows"].delete_many(
                        {"dataset": kunci, "jenis": jenis, "sumber": {"$in": SUMBER_TAMBAHAN}})
                    db["datasets"].update_one({"dataset": kunci, "jenis": jenis}, {"$set": {"jumlah_tambahan": 0}})
                    _naik_versi("tam_ver")
                    st.session_state[f"{kp}_pesan"] = "Data tambahan dihapus."
                    st.rerun()
                except Exception as e:
                    st.error(f"Gagal menghapus: {e}")


# ---------------------------------------------------------------- unggah dataset baru
def proses_csv(df, mode, target):
    """
    Bersihkan dan periksa CSV. mode: "angka" (regresi), "kelas" (klasifikasi) atau None (clustering).
    Mengembalikan (df atau None, daftar_pesan, ada_galat). Kolom target diletakkan di paling kanan.
    """
    pesan = []
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()].dropna(axis=1, how="all")
    if len(df) < MIN_BARIS:
        return None, [f"Minimal {MIN_BARIS} baris data."], True
    if len(df) > MAKS_BARIS_DATASET:
        return None, [f"Maksimal {MAKS_BARIS_DATASET} baris per dataset."], True
    if df.shape[1] > MAKS_KOLOM:
        return None, [f"Maksimal {MAKS_KOLOM} kolom."], True

    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        ada = int(df[c].notna().sum())
        if ada and int(s.notna().sum()) >= 0.9 * ada:
            df[c] = s
        else:
            df[c] = df[c].map(lambda v: None if pd.isna(v) else str(v).strip())

    buang = [c for c in df.columns if c != target and not _numerik(df[c]) and df[c].nunique() > 50]
    if buang:
        df = df.drop(columns=buang)
        pesan.append("Kolom teks dengan terlalu banyak nilai berbeda diabaikan: " + ", ".join(buang))

    if mode == "angka":
        if not _numerik(df[target]):
            return None, [f"Kolom target '{target}' harus berisi angka untuk regresi."], True
        df = df[df[target].notna()]
    elif mode == "kelas":
        df = df[df[target].notna()]
        vc = df[target].value_counts()
        if vc.size > 20:
            return None, [f"Kolom target punya {vc.size} nilai berbeda. Klasifikasi butuh 2 sampai 20 kelas."], True
        kecil = vc[vc < 2].index
        if len(kecil):
            df = df[~df[target].isin(kecil)]
            pesan.append("Kelas dengan kurang dari 2 baris dibuang: " + ", ".join(map(str, kecil)))
        if df[target].nunique() < 2:
            return None, ["Klasifikasi butuh minimal 2 kelas dengan masing-masing 2 baris atau lebih."], True

    fitur = [c for c in df.columns if c != target]
    if not fitur:
        return None, ["Tidak ada kolom fitur yang bisa dipakai."], True
    df = df[fitur + ([target] if target else [])].reset_index(drop=True)
    if len(df) < MIN_BARIS:
        return None, [f"Setelah dibersihkan, data kurang dari {MIN_BARIS} baris."], True
    return df, pesan, False


def hapus_dataset(kunci, jenis):
    db = get_db()
    db["dataset_rows"].delete_many({"dataset": kunci, "jenis": jenis})
    db["datasets"].delete_many({"dataset": kunci, "jenis": jenis})
    _naik_versi()


def render_unggah_dataset(jenis, kp, mode, nama_dilarang, pilihan, pilihan_awal):
    """Unggah CSV sebagai dataset baru. Setelah tersimpan, dataset muncul di dropdown."""
    st.write("Unggah CSV untuk membuat **dataset baru**. Dataset ini tersimpan di MongoDB dan muncul di "
             "dropdown Dataset di tab Data.")
    pesan = st.session_state.pop(f"{kp}_pesan_unggah", None)
    if pesan:
        st.success(pesan)

    n = st.session_state.get(f"{kp}_n", 0)
    nama = st.text_input("Nama dataset", max_chars=40, key=f"{kp}_nama_{n}").strip()
    berkas = st.file_uploader("File CSV", type="csv", key=f"{kp}_up_{n}")
    if berkas is not None:
        try:
            try:
                mentah = pd.read_csv(berkas, on_bad_lines="skip")
            except UnicodeDecodeError:
                berkas.seek(0)
                mentah = pd.read_csv(berkas, on_bad_lines="skip", encoding="latin-1")
        except Exception as e:
            st.error(f"CSV tidak bisa dibaca: {e}")
            return
        target = None
        if mode:
            kolom = [str(c).strip() for c in mentah.columns]
            target = st.selectbox("Kolom target", kolom, index=len(kolom) - 1, key=f"{kp}_tgt_{n}")
        bersih, info, galat = proses_csv(mentah, mode, target)
        for p in info:
            (st.error if galat else st.warning)(p)
        if bersih is None:
            return
        st.write(f"{len(bersih)} baris dan {bersih.shape[1]} kolom siap disimpan. Pratinjau:")
        st.dataframe(bersih.head(10), use_container_width=True)
        if len(nama) < 3:
            st.info("Isi nama dataset (minimal 3 karakter) untuk menyimpan.")
        elif nama in nama_dilarang:
            st.error("Nama itu sudah dipakai. Pilih nama lain.")
        elif st.button("Simpan sebagai dataset baru", type="primary", key=f"{kp}_simpan"):
            try:
                if len(daftar_unggahan(jenis, _versi())) >= MAKS_DATASET_UNGGAHAN:
                    st.error(f"Dataset unggahan dibatasi {MAKS_DATASET_UNGGAHAN} per halaman. Hapus salah satu dulu.")
                else:
                    salin_ke_mongo(get_db(), nama, jenis, bersih, [target] if target else [], "unggahan", berkas.name)
                    _naik_versi()
                    st.session_state[f"{kp}_n"] = n + 1
                    st.session_state[f"pilih_next_{jenis}"] = nama
                    st.session_state[f"{kp}_pesan_unggah"] = f"Dataset '{nama}' tersimpan dan sudah dipilih di tab Data."
                    st.rerun()
            except Exception as e:
                st.error(f"Gagal menyimpan: {e}")

    if pilihan in daftar_unggahan(jenis, _versi()):
        with st.expander(f"Hapus dataset '{pilihan}'"):
            yakin = st.checkbox("Saya yakin, data dan tambahan retrainingnya ikut terhapus", key=f"{kp}_yakin_ds")
            if st.button("Hapus dataset ini", disabled=not yakin, key=f"{kp}_hapus_ds"):
                try:
                    hapus_dataset(pilihan, jenis)
                    st.session_state[f"pilih_next_{jenis}"] = pilihan_awal
                    st.session_state[f"{kp}_pesan_unggah"] = f"Dataset '{pilihan}' dihapus."
                    st.rerun()
                except Exception as e:
                    st.error(f"Gagal menghapus: {e}")


# ---------------------------------------------------------------- umum
def catat_eksperimen(dokumen):
    dokumen = dict(dokumen)
    dokumen["waktu"] = datetime.now(timezone.utc)
    get_db()["eksperimen"].insert_one(dokumen)


def tampilkan_perbandingan(sebelum, sesudah, metrik, lebih_kecil=(), netral=()):
    kolom = st.columns(len(metrik))
    for kol, m in zip(kolom, metrik):
        a, b = sebelum.get(m), sesudah.get(m)
        if a is None or b is None:
            kol.metric(m, "-")
            continue
        warna = "off" if m in netral else ("inverse" if m in lebih_kecil else "normal")
        kol.metric(m, f"{b:.4f}", delta=f"{b - a:+.4f}", delta_color=warna,
                   help=f"Sebelum: {a:.4f}")