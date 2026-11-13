import streamlit as st


TEMA = {
    "home":        ("#0b1830", "#161B26", "#4DA3FF"),
    "regresi":     ("#0b1830", "#161B26", "#2ED3A0"),
    "klasifikasi": ("#0b1830", "#161B26", "#FF8A5B"),
    "clustering":  ("#0b1830", "#161B26", "#B18CFF"),
    "CNN":         ("#0b1830", "#161B26", "#A3E635"),
    "komentar":    ("#0b1830", "#161B26", "#FFD166"),
    "ai":          ("#0b1830", "#161B26", "#F472B6"),
}

def terapkan_tema(nama):
    """Panggil di baris awal tiap halaman: terapkan_tema("regresi")."""
    awal, akhir, aksen = TEMA.get(nama, TEMA["home"])
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(135deg, {awal} 0%, {akhir} 100%);
            background-attachment: fixed;
        }}
        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stSidebar"] {{
            background-color: rgba(0, 0, 0, 0.35);
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }}
        h1, h2, h3 {{ letter-spacing: -0.01em; }}
        h1 {{ border-bottom: 3px solid {aksen}; padding-bottom: 0.3rem; display: inline-block; }}
        [data-testid="stMetric"] {{
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 12px 16px;
        }}
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: rgba(255, 255, 255, 0.03);
            border-radius: 12px;
        }}
        .stButton > button[kind="primary"] {{
            background-color: {aksen};
            border: none;
            color: #0b0b12;
            font-weight: 600;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )