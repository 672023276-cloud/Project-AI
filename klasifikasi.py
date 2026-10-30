import streamlit as st

st.text("KLASIFIKASI")


pilihanModel = st.selectbox("Pilih Algoritma Clustering", ["K-Means", "DBSCAN", "Hierarchical"]) 

if pilihanModel == "K-Means":
    st.write("Anda memilih algoritma K-Means")
    # Tambahkan kode untuk K-Means di sini
elif pilihanModel == "DBSCAN":
    st.write("Anda memilih algoritma DBSCAN")
    # Tambahkan kode untuk DBSCAN di sini
elif pilihanModel == "Hierarchical":
    st.write("Anda memilih algoritma Hierarchical")
    # Tambahkan kode untuk Hierarchical di sini     
    
pilihandata = st.selectbox("Pilih Algoritma Clustering", ["A", "B", "C"]) 

if pilihandata == "A":
    st.write("Anda memilih A")
    # Tambahkan kode untuk K-Means di sini
elif pilihandata == "B":
    st.write("Anda memilih B")
    # Tambahkan kode untuk DBSCAN di sini
elif pilihandata == "C":
    st.write("Anda memilih C")
    # Tambahkan kode untuk Hierarchical di sini     
    
    
    
    