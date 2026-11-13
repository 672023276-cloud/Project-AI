import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime
import db

st.title("🌤️ Real-Time Weather Streamer (Open-Meteo)")
st.write("Mengambil data cuaca asli secara real-time dan menyimpannya langsung ke MongoDB Atlas.")

# Pilihan kota (Koordinat latitude & longitude otomatis)
kota_pilihan = st.selectbox(
    "Pilih Lokasi Pemantauan Cuaca:",
    ["Salatiga", "Purwodadi", "Semarang", "Jakarta", "Yogyakarta"]
)

# Koordinat GPS berdasarkan pilihan kota
koordinat = {
    "Salatiga": {"lat": -7.3305, "lon": 110.5084},
    "Purwodadi": {"lat": -7.0863, "lon": 110.9156},
    "Semarang": {"lat": -6.9666, "lon": 110.4194},
    "Jakarta": {"lat": -6.2088, "lon": 106.8456},
    "Yogyakarta": {"lat": -7.7956, "lon": 110.3695}
}

lat = koordinat[kota_pilihan]["lat"]
lon = koordinat[kota_pilihan]["lon"]

col1, col2 = st.columns(2)
with col1:
    interval = st.number_input("Interval Ambil Data (detik)", min_value=5, max_value=60, value=10)
with col2:
    max_loop = st.number_input("Jumlah Pengambilan Data", min_value=1, max_value=20, value=5)

mulai = st.button("▶️ Mulai Ambil Data Cuaca Real-Time", type="primary")

status_text = st.empty()
chart_placeholder = st.empty()
table_placeholder = st.empty()

if mulai:
    try:
        database = db.get_db() if hasattr(db, "get_db") else db.db
        collection = database["realtime_weather_data"]
    except Exception as e:
        st.error(f"Gagal mengakses database: {e}")
        st.stop()

    buffer_data = []
    api_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m"

    for i in range(int(max_loop)):
        try:
            # Ambil data asli dari internet via Open-Meteo API
            response = requests.get(api_url)
            result = response.json()
            current = result["current"]

            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "kota": kota_pilihan,
                "suhu_c": current["temperature_2m"],
                "kelembaban_persen": current["relative_humidity_2m"],
                "kecepatan_angin_ms": current["wind_speed_10m"]
            }

            # Simpan permanen ke MongoDB Atlas
            collection.insert_one(record.copy())

            # Tampilkan ke Streamlit secara live
            buffer_data.append(record)
            df_live = pd.DataFrame(buffer_data)

            status_text.success(f"[{i+1}/{max_loop}] Berhasil mengambil data cuaca real-time untuk **{kota_pilihan}** dan menyimpannya ke MongoDB!")
            chart_placeholder.line_chart(df_live.set_index("timestamp")[["suhu_c", "kelembaban_persen"]])
            table_placeholder.dataframe(df_live.iloc[::-1], use_container_width=True)

        except Exception as err:
            st.error(f"Gagal mengambil data dari API: {err}")
            break

        if i < int(max_loop) - 1:
            time.sleep(interval)

    st.balloons()
    st.info("Pengambilan data cuaca selesai dan tersimpan aman di database!")