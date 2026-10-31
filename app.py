import streamlit as st

st.set_page_config(page_title="TAS ML App", layout="wide")

st.set_page_config(initial_sidebar_state='collapsed')

pg = st.navigation([
    st.Page("home.py", title="Home", icon="🏠", default=True),
    st.Page("regresi.py", title="Regresi", icon="📈"),
    st.Page("klasifikasi.py", title="Klasifikasi", icon="🎯"),
    st.Page("clustering.py", title="Clustering", icon="🔵"),
])
pg.run()






