import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import joblib

# ------------------------
# Load data & model
# ------------------------
df_raw_base = pd.read_csv("data_gabungan_lengkap.csv")
df_2026_base = pd.read_csv("New_overscore_all.csv")

MODEL_FILENAME = "model_2026.pkl"
MODEL_PATH = os.path.join(os.getcwd(), MODEL_FILENAME)

score_cols = [col for col in df_raw_base.columns if "overall_score_" in col]

# Transform ke long format
df_long_base = pd.melt(
    df_raw_base,
    id_vars=[col for col in df_raw_base.columns if not str(col).startswith("overall_score")],
    value_vars=score_cols,
    var_name="year",
    value_name="overall_score"
)
df_long_base["year"] = df_long_base["year"].str.extract(r"(\d{4})")[0].astype(float).dropna().astype(int)

# Load model
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Gagal load model: {str(e)}")

# ------------------------
# Sidebar & Upload
# ------------------------
st.sidebar.header("Menu Navigasi & Upload CSV")
menu = st.sidebar.radio("Pilih Menu", ["Home", "Dashboard", "Prediksi", "Pergeseran Peringkat", "Tampilan Dataset"])
uploaded_file = st.sidebar.file_uploader("Upload dataset tambahan (opsional)", type=["csv"])

# Inisialisasi dataset yang akan digunakan
df_raw = df_raw_base.copy()
df_long = df_long_base.copy()
df_2026 = df_2026_base.copy()

if uploaded_file:
    try:
        df_uploaded = pd.read_csv(uploaded_file)

        required_cols = ["institution", "overall_score"]
        optional_cols = ["academic_reputation_score", "employer_reputation_score", 
                         "faculty_student_score", "citations_score", 
                         "international_faculty_score", "international_student_score", 
                         "region", "country"]

        # Cek kolom wajib
        missing_required = [col for col in required_cols if col not in df_uploaded.columns]
        if missing_required:
            st.sidebar.error(f"❌ Kolom wajib hilang: {', '.join(missing_required)}")
        else:
            # Info kolom optional
            available_optional = [col for col in optional_cols if col in df_uploaded.columns]
            missing_optional = [col for col in optional_cols if col not in df_uploaded.columns]

            st.sidebar.success(f"✅ {df_uploaded.shape[0]} baris dimuat.")
            if available_optional:
                st.sidebar.info(f"Kolom optional tersedia: {', '.join(available_optional)}")
            if missing_optional:
                st.sidebar.info(f"Kolom optional hilang: {', '.join(missing_optional)}")

            # Tambahkan kolom tahun dan overall_score_2026
            df_uploaded["overall_score_2026"] = df_uploaded["overall_score"]
            df_uploaded["year"] = 2026

            # Gabungkan
            df_raw = pd.concat([df_raw, df_uploaded], ignore_index=True)

            # Update long format
            df_long_uploaded = pd.melt(
                df_uploaded,
                id_vars=[col for col in df_uploaded.columns if not str(col).startswith("overall_score")],
                value_vars=["overall_score_2026"],
                var_name="year",
                value_name="overall_score"
            )
            df_long_uploaded["year"] = 2026
            df_long = pd.concat([df_long, df_long_uploaded], ignore_index=True)

            # Update df_2026
            df_2026 = pd.concat([df_2026, df_uploaded[["institution", "overall_score_2026"] + available_optional]], ignore_index=True)

    except Exception as e:
        st.sidebar.error(f"⚠️ Gagal membaca file: {e}")

# ------------------------
# MENU HOME
# ------------------------
if menu == "Home":
    st.title("🌐 University Rankings & Performance Dashboard")
    st.image("Kampus.png", use_column_width=True, caption="Ilustrasi Kampus dan Peringkat")
   # Penjelasan di bagian atas home
    st.markdown("""
    ## Latar Belakang  
    Aplikasi ini dikembangkan untuk menganalisis dan memprediksi peringkat universitas global berdasarkan indikator kinerja akademik dan riset.  
    Hasil analisis mendukung pengambilan keputusan strategis dalam penyusunan program beasiswa, kolaborasi, outsourcing, serta pengembangan talenta masa depan.  

    ## Tujuan  
    - Menyusun prediksi peringkat universitas global untuk 2025 dan 2026.  
    - Menggali insight utama untuk mendukung strategi kolaborasi dan rekrutmen.  

    ## Sumber & Transformasi Data  
    Data berasal dari peringkat universitas global tahun 2018, 2019, 2021, 2023, 2024, dan 2025.  
    Data memuat skor keseluruhan, reputasi akademik & pemberi kerja, rasio dosen-mahasiswa, intensitas riset, hingga keberadaan mahasiswa/staf internasional.  
    Data terbaru digunakan untuk validasi & proyeksi.  

    ## Indikator Penilaian  
    - *Academic Reputation Score:* Survei global terhadap akademisi tentang kualitas institusi.  
    - *Employer Reputation Score:* Survei pemberi kerja terkait lulusan terbaik.  
    - *Citations per Faculty:* Jumlah sitasi publikasi ilmiah dibagi jumlah staf pengajar.  
    - *Faculty Student Score:* Rasio jumlah staf pengajar terhadap mahasiswa.  
    """)


# ------------------------
# DASHBOARD
# ------------------------
elif menu == "Dashboard":
    st.title("📊 Dashboard Peringkat Universitas")
    year = st.sidebar.selectbox("Pilih Tahun", sorted(df_long["year"].unique(), reverse=True))
    df_year = df_long[df_long["year"] == year].copy()
    df_year = df_year.dropna(subset=["overall_score"])
    df_year["Rank"] = df_year["overall_score"].rank(ascending=False, method="min").astype(int)

    col1, col2 = st.columns(2)
    col1.metric("Jumlah Universitas", df_year["institution"].nunique())
    col2.metric("Rata-rata Overall Score", round(df_year["overall_score"].mean(), 2))

    st.subheader(f"🏆 Top 10 Universitas Tahun {year}")
    top10 = df_year.sort_values(by="overall_score", ascending=False).head(10)
    fig = px.bar(top10, x="overall_score", y="institution", orientation="h",
                 color="overall_score", color_continuous_scale="viridis")
    fig.update_layout(yaxis=dict(categoryorder='total ascending'))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Tren Rata-rata Overall Score")
    avg_score = df_long.groupby("year")["overall_score"].mean().reset_index()
    fig2 = px.line(avg_score, x="year", y="overall_score", markers=True)
    st.plotly_chart(fig2, use_container_width=True)

# ------------------------
# PERGESERAN
# ------------------------
elif menu == "Pergeseran Peringkat":
    st.title("📊 Pergeseran Peringkat Top 10")
    pilihan = st.selectbox("Tahun", sorted(df_long["year"].unique()))
    df_top = df_long[df_long["year"] == pilihan].sort_values(by="overall_score", ascending=False).head(10)

    fig = px.bar(df_top, x="overall_score", y="institution", orientation="h",
                 color="overall_score", color_continuous_scale="blues",
                 title=f"Top 10 Tahun {pilihan}")
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig, use_container_width=True)

# ------------------------
# DATASET
# ------------------------
elif menu == "Tampilan Dataset":
    st.title("📄 Dataset Peringkat Universitas")
    years = st.sidebar.multiselect("Pilih Tahun", sorted(df_long["year"].unique()), default=sorted(df_long["year"].unique()))
    df_filtered = df_long[df_long["year"].isin(years)]
    st.dataframe(df_filtered, use_container_width=True)
    csv = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Unduh CSV", csv, file_name="filtered_data.csv", mime="text/csv")

# ------------------------
# PREDIKSI (tidak diubah, tetap seperti kode lo)
# ------------------------
elif menu == "Prediksi":
    st.title("🧠 Prediksi Overall Score Universitas Baru")
    if model is None:
        st.error("Model belum tersedia. Pastikan model_2026.pkl ada di direktori.")
    else:
        with st.form("form_prediksi"):
            st.subheader("📥 Masukkan Nilai Fitur")
            nama_kampus = st.text_input("Nama Universitas (contoh: ITB / dummy)")
            col1, col2 = st.columns(2)
            with col1:
                academic = st.number_input("Academic Reputation Score (0-100)", 0.0, 100.0)
                employer = st.number_input("Employer Reputation Score (0-100)", 0.0, 100.0)
            with col2:
                citations = st.number_input("Citations per Faculty (0-100)", 0.0, 100.0)
                faculty_student = st.number_input("Faculty Student Score (0-100)", 0.0, 100.0)
            submitted = st.form_submit_button("🔮 Prediksi Skor")
            if submitted:
                if nama_kampus.strip() == "":
                    st.warning("Masukkan nama universitas.")
                else:
                    fitur = np.array([[academic, employer, citations, faculty_student]])
                    prediksi = model.predict(fitur)[0]
                    df_temp = pd.concat([df_2026, pd.DataFrame({
                        "institution": [nama_kampus],
                        "overall_score_2026": [prediksi]
                    })], ignore_index=True)
                    df_temp["rank_prediksi"] = df_temp["overall_score_2026"].rank(ascending=False, method="min").astype(int)
                    rank_pred = df_temp.loc[df_temp["institution"] == nama_kampus, "rank_prediksi"].values[0]
                    st.success(f"🎯 Prediksi Overall Score: {prediksi:.2f}")
                    st.info(f"🏅 Perkiraan Peringkat: #{rank_pred} dari {len(df_temp)} universitas")
