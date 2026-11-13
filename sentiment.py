import streamlit as st
from groq import Groq
from db import get_db, cek_koneksi

st.subheader("📊 Analisis Sentimen Komentar & Teks", divider="rainbow", anchor=False)
st.write("Analisis sentimen otomatis menggunakan LLM Groq (bisa dari input bebas atau langsung dari database komentar).")

# Konfigurasi model Groq
MODEL_CONFIG = {
    "openai/gpt-oss-120b": {
        "name": "GPT OSS 120B",
        "tokens": 65536,
        "developer": "OpenAI"
    },

    "openai/gpt-oss-20b": {
        "name": "GPT OSS 20B",
        "tokens": 65536,
        "developer": "OpenAI"
    },
}

selected_model = st.selectbox("Pilih Model AI:", options=list(MODEL_CONFIG.keys()), format_func=lambda x: MODEL_CONFIG[x]["name"])
model_info = MODEL_CONFIG[selected_model]

st.caption(f"Model aktif: **{model_info['name']}** ({model_info['developer']}) • Max context: {model_info['tokens']:,} tokens")

# Ambil API key dari st.secrets
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except:
    groq_api_key = ""

if not groq_api_key:
    groq_api_key = st.text_input("Masukkan Groq API Key:", type="password")

st.divider()

# Pilih mode sumber teks
sumber_teks = st.radio("Pilih Sumber Teks yang Ingin Diansalisis:", ["Tulis Manual / Ulasan Baru", "Ambil dari Database Komentar Publik"])

teks_ulasan = ""

if sumber_teks == "Tulis Manual / Ulasan Baru":
    teks_ulasan = st.text_area(
        "Tulis teks / ulasan di sini:",
        placeholder="Contoh: Aplikasi ini sangat membantu dan luar biasa cepat kinerjanya!",
        height=130
    )
else:
    # Cek koneksi dan ambil data komentar dari MongoDB
    ok, pesan = cek_koneksi()
    if not ok:
        st.error(pesan)
    else:
        try:
            koleksi = get_db()["komentar"]
            daftar_komentar = list(koleksi.find().sort("waktu", -1).limit(50))
            
            if not daftar_komentar:
                st.info("Belum ada komentar di database.")
            else:
                # Buat pilihan dropdown berdasarkan nama dan isi komentar
                pilihan_map = {f"{k.get('nama', 'Anonim')}: {k.get('komentar', '')[:50]}...": k.get('komentar') for k in daftar_komentar}
                komentar_terpilih = st.selectbox("Pilih komentar dari database:", options=list(pilihan_map.keys()))
                
                if komentar_terpilih:
                    teks_ulasan = pilihan_map[komentar_terpilih]
                    st.info(f"**Teks yang dipilih:** \"{teks_ulasan}\"")
        except Exception as e:
            st.error(f"Gagal memuat daftar komentar: {e}")

st.divider()

# Tombol Eksekusi Analisis
if st.button("🔍 Analisis Sentimen Sekarang", type="primary"):
    if not teks_ulasan.strip():
        st.warning("Mohon masukkan atau pilih teks terlebih dahulu!")
    elif not groq_api_key:
        st.error("Groq API Key belum diatur di secrets atau input!")
    else:
        try:
            client = Groq(api_key=groq_api_key)
            
            system_prompt = (
                "Anda adalah sistem analisis sentimen yang andal. "
                "Analisis teks berikut dan berikan hasil dengan format:\n"
                "1. **Sentimen**: (Positif / Negatif / Netral)\n"
                "2. **Alasan / Penjelasan Singkat**."
            )

            with st.spinner(f"Sedang menganalisis sentimen dengan {model_info['name']}..."):
                response = client.chat.completions.create(
                    model=selected_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": teks_ulasan}
                    ],
                    max_tokens=2048
                )
                hasil_sentimen = response.choices[0].message.content

            # Visualisasi hasil berdasarkan teks sentimen
           # Cek hanya pada baris pertama hasil AI untuk menentukan warna secara akurat
            baris_pertama = hasil_sentimen.split('\n')[0].lower()
            
            if "positif" in baris_pertama:
                st.success("✨ Analisis Selesai: Sentimen Terdeteksi POSITIF!")
            elif "negatif" in baris_pertama:
                st.error("⚠️ Analisis Selesai: Sentimen Terdeteksi NEGATIF!")
            else:
                st.info("ℹ️ Analisis Selesai: Sentimen Terdeteksi NETRAL!")
            st.markdown("### Hasil Analisis AI:")
            st.markdown(hasil_sentimen)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses sentimen: {e}")