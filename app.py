import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Judul Aplikasi
st.title("Aplikasi Prediksi Gaji Pertama Peserta Vokasi")
st.write("Aplikasi ini memprediksi gaji pertama (dalam juta) menggunakan model Linear Regression.")

# 2. Fungsi untuk memuat resource
@st.cache_resource
def load_resources():
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('scaler_features.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('linear_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return encoders, scaler, model

try:
    encoders, scaler, model = load_resources()

    # 3. Input Form
    st.sidebar.header("Input Fitur")
    
    jenis_kelamin = st.sidebar.selectbox("Jenis Kelamin", options=['L', 'P'])
    usia = st.sidebar.number_input("Usia", min_value=17, max_value=50, value=25)
    pendidikan = st.sidebar.selectbox("Pendidikan", options=['SMA', 'SMK', 'D3', 'S1'])
    jurusan = st.sidebar.selectbox("Jurusan", options=['otomotif', 'desain grafis', 'teknik las', 'teknik listrik', 'administrasi'])
    durasi_jam = st.sidebar.number_input("Durasi Jam Pelatihan", min_value=30, max_value=100, value=60)
    nilai_ujian = st.sidebar.number_input("Nilai Ujian", min_value=0.0, max_value=100.0, value=85.0)
    status_bekerja = st.sidebar.selectbox("Status Bekerja", options=['Belum Bekerja', 'Sudah Bekerja'])

    # 4. Tombol Prediksi
    if st.button("Prediksi Gaji"):
        # Menyusun data input
        input_data = pd.DataFrame([{
            'Jenis_Kelamin': jenis_kelamin,
            'Usia': usia,
            'Pendidikan': pendidikan,
            'Jurusan': jurusan,
            'Durasi_Jam': durasi_jam,
            'Nilai_Ujian': nilai_ujian,
            'Status_Bekerja': status_bekerja
        }])

        # Preprocessing: Label Encoding
        for col, le in encoders.items():
            input_data[col] = le.transform(input_data[col])

        # Preprocessing: Scaling
        input_scaled = scaler.transform(input_data)

        # Melakukan Prediksi
        prediction = model.predict(input_scaled)

        # Menampilkan Hasil
        st.success(f"Estimasi Gaji Pertama Anda adalah: Rp {prediction[0]:.2f} Juta")

except FileNotFoundError:
    st.error("File pendukung (model/scaler/encoder) tidak ditemukan. Pastikan sudah melakukan export di notebook.")
