import streamlit as st

st.set_page_config(page_title="TAS ML App", layout="wide")

st.set_page_config(initial_sidebar_state='expanded')

pg = st.navigation([
    st.Page("home.py", title="Home", icon=":material/home:", default=True),
    st.Page("regresi.py", title="Regresi", icon=":material/show_chart:"),
    st.Page("klasifikasi.py", title="Klasifikasi", icon=":material/target:"),
    st.Page("clustering.py", title="Clustering", icon=":material/hub:"),
    st.Page("komentar.py", title="Komentar", icon=":material/chat:"),
    st.Page("ai.py", title="groq", icon=":material/smart_toy:"),
    st.Page("CNN.py", title="Deep Learning CNN", icon=":material/auto_awesome:"),
])
pg.run()






