import streamlit as st
from pymongo import MongoClient


@st.cache_resource
def get_client():
    return MongoClient(st.secrets["MONGO_URI"], serverSelectionTimeoutMS=5000)


def get_db():
    return get_client()["tas_db"]


def cek_koneksi():
    """Return (berhasil: bool, pesan: str)."""
    try:
        get_client().admin.command("ping")
        return True, "Terhubung ke MongoDB Atlas"
    except Exception as e:
        return False, f"Gagal terhubung: {e}"


def hitung(nama_koleksi):
    """Jumlah dokumen di sebuah koleksi, 0 jika gagal."""
    try:
        return get_db()[nama_koleksi].count_documents({})
    except Exception:
        return 0