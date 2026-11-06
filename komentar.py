
import streamlit as st
from datetime import datetime, timezone, timedelta
from db import get_db, cek_koneksi

WIB = timezone(timedelta(hours=7))
KOLEKSI = "komentar"
MAKS_NAMA = 50
MAKS_KOMENTAR = 500

st.title("Komentar Publik")
st.caption("Tinggalkan komentar tentang aplikasi ini. Semua komentar tersimpan di MongoDB Atlas.")

ok, pesan = cek_koneksi()
if not ok:
    st.error(pesan)
    st.stop()

koleksi = get_db()[KOLEKSI]

# --- Form input ---
st.subheader("Tulis Komentar")
with st.form("form_komentar", clear_on_submit=True):
    nama = st.text_input("Nama", max_chars=MAKS_NAMA, placeholder="Nama kamu")
    isi = st.text_area(
        "Komentar",
        max_chars=MAKS_KOMENTAR,
        placeholder="Tulis komentar di sini...",
        height=120,
    )
    kirim = st.form_submit_button("Kirim Komentar", type="primary")

if kirim:
    nama = nama.strip()
    isi = isi.strip()
    if not nama or not isi:
        st.warning("Nama dan komentar tidak boleh kosong.")
    else:
        try:
            koleksi.insert_one({
                "nama": nama,
                "komentar": isi,
                "waktu": datetime.now(timezone.utc),
            })
            st.success("Komentar berhasil disimpan ke MongoDB.")
        except Exception as e:
            st.error(f"Gagal menyimpan komentar: {e}")

st.divider()

# --- Daftar komentar ---
st.subheader("Komentar Terbaru")

try:
    total = koleksi.count_documents({})
    jumlah = st.slider("Tampilkan komentar", 5, 50, 10, step=5)
    daftar = list(koleksi.find().sort("waktu", -1).limit(jumlah))
except Exception as e:
    st.error(f"Gagal memuat komentar: {e}")
    st.stop()

st.caption(f"Total komentar: {total}")

if not daftar:
    st.info("Belum ada komentar. Jadilah yang pertama.")
else:
    for k in daftar:
        waktu = k.get("waktu")
        if isinstance(waktu, datetime):
            if waktu.tzinfo is None:
                waktu = waktu.replace(tzinfo=timezone.utc)
            teks_waktu = waktu.astimezone(WIB).strftime("%d %b %Y, %H:%M WIB")
        else:
            teks_waktu = "-"
        with st.container(border=True):
            st.markdown(f"**{k.get('nama', 'Anonim')}**  ·  {teks_waktu}")
            st.text(k.get("komentar", ""))