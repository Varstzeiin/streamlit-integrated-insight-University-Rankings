# app_unified.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import joblib

# ------------------------
# Load data & model
# ------------------------
df_raw = pd.read_csv("data_gabungan_lengkap.csv")
df_2026 = pd.read_csv("New_overscore_all.csv")
MODEL_FILENAME = "model_2026.pkl"
MODEL_PATH = os.path.join(os.getcwd(), MODEL_FILENAME)

score_cols = [col for col in df_raw.columns if "overall_score_" in col]
df_long = pd.melt(
    df_raw,
    id_vars=[
        'institution', 'location', 'region', 'classification', 'focus', 'research_intensity',
        'academic_reputation_score', 'academic_reputation_rank', 'employer_reputation_score', 'employer_reputation_rank',
        'faculty_student_score', 'faculty_student_rank', 'citations_score', 'citations_per_faculty_rank',
        'international_faculty_score', 'international_faculty_rank', 'international_student_score', 'international_students_rank',
        'international_research_network_score', 'international_research_network_rank', 'employment_outcomes_score', 'employment_outcomes_rank',
        'sustainability_score', 'sustainability_rank', 'status_A', 'status_B', 'status_C'
    ],
    value_vars=score_cols,
    var_name="year",
    value_name="overall_score"
)

# Perbaikan ekstrak tahun
df_long["year"] = df_long["year"].str.extract(r"(\d{4})")[0]
df_long = df_long.dropna(subset=["year"])
df_long["year"] = df_long["year"].astype(int)

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

if uploaded_file:
    try:
        df_uploaded = pd.read_csv(uploaded_file)

        # Kolom wajib & optional
        required_cols = ["institution", "overall_score_2026"]
        optional_cols = [
            "academic_reputation_score", "employer_reputation_score",
            "faculty_student_score", "citations_score",
            "international_faculty_score", "international_student_score"
        ]

        # Cek kolom wajib
        missing_required = [col for col in required_cols if col not in df_uploaded.columns]
        if missing_required:
            st.sidebar.error(f"❌ File wajib punya kolom: {', '.join(missing_required)}")
        else:
            # Cek optional
            available_optional = [col for col in optional_cols if col in df_uploaded.columns]
            missing_optional = [col for col in optional_cols if col not in df_uploaded.columns]

            st.sidebar.success(f"✅ Data tambahan dimuat: {df_uploaded.shape[0]} baris.")
            if available_optional:
                st.sidebar.info(f"✅ Kolom optional tersedia: {', '.join(available_optional)}")
            if missing_optional:
                st.sidebar.info(f"ℹ️ Kolom optional tidak ada: {', '.join(missing_optional)}")

            # Proses data
            df_uploaded["year"] = 2026

            # Siapkan df_long
            df_long = pd.melt(
                df_uploaded,
                id_vars=[col for col in df_uploaded.columns if not str(col).startswith("overall_score")],
                value_vars=[col for col in df_uploaded.columns if str(col).startswith("overall_score")],
                var_name="year",
                value_name="overall_score"
            )

            df_long["year"] = df_long["year"].str.extract(r"(\d{4})")
            df_long["year"] = df_long["year"].fillna(2026).astype(int)

            # Update df_2026
            df_2026 = df_uploaded[["institution", "overall_score_2026"]]

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
    - **Academic Reputation Score:** Survei global terhadap akademisi tentang kualitas institusi.  
    - **Employer Reputation Score:** Survei pemberi kerja terkait lulusan terbaik.  
    - **Citations per Faculty:** Jumlah sitasi publikasi ilmiah dibagi jumlah staf pengajar.  
    - **Faculty Student Score:** Rasio jumlah staf pengajar terhadap mahasiswa.  
    """)
# ------------------------
# MENU DASHBOARD
# ------------------------
elif menu == "Dashboard":
    st.title("📊 Dashboard Peringkat Universitas")
    st.sidebar.header("Filter Tahun")
    year = st.sidebar.selectbox("Pilih Tahun", sorted(df_long["year"].unique(), reverse=True))

    df_year = df_long[df_long["year"] == year].copy()
    df_year = df_year.sort_values(by="overall_score", ascending=False).reset_index(drop=True)
    df_year["Rank"] = df_year["overall_score"].rank(ascending=False, method="min").astype(int)

    col1, col2 = st.columns(2)
    col1.metric("Jumlah Universitas", df_year["institution"].nunique())
    col2.metric("Rata-rata Overall Score", round(df_year["overall_score"].mean(), 2))

    st.subheader(f"🏆 Top 10 Universitas Tahun {year}")
    top10 = df_year.head(10)
    fig = px.bar(top10, x="overall_score", y="institution", orientation="h",
                 color="overall_score", color_continuous_scale="viridis",
                 title=f"10 Besar Universitas Tahun {year}")
    fig.update_layout(yaxis=dict(categoryorder='total ascending'))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Tren Rata-rata Overall Score (2018–2026)")
    avg_score_2026 = df_2026["overall_score_2026"].mean()
    avg_score = df_long.groupby("year")["overall_score"].mean().reset_index()
    avg_score = pd.concat([avg_score, pd.DataFrame({"year": [2026], "overall_score": [avg_score_2026]})], ignore_index=True)
    fig2 = px.line(avg_score, x="year", y="overall_score", markers=True, title="Tren Rata-rata Overall Score per Tahun")
    st.plotly_chart(fig2, use_container_width=True)

# ------------------------
# MENU PREDIKSI
# ------------------------
elif menu == "Prediksi":
    st.title("🧠 Prediksi Overall Score Universitas Baru")

    if model is None:
        st.error("Model belum tersedia. Pastikan model_2026.pkl berada di direktori.")
    else:
        with st.form("form_prediksi"):
            st.subheader("📥 Masukkan Nilai Fitur:")

            nama_kampus = st.text_input("Nama Universitas (contoh: ITB / dummy)")

            col1, col2 = st.columns(2)
            with col1:
                academic = st.number_input("Academic Reputation Score (Range 0-100)", 0.0, 100.0, step=0.1)
                employer = st.number_input("Employer Reputation Score (Range 0-100)", 0.0, 100.0, step=0.1)
            with col2:
                citations = st.number_input("Citations per Faculty (Range 0-100)", 0.0, 100.0, step=0.1)
                faculty_student = st.number_input("Faculty Student Score (Range 0-100)", 0.0, 100.0, step=0.1)

            submitted = st.form_submit_button("🔮 Prediksi Skor")

            if submitted:
                if nama_kampus.strip() == "":
                    st.warning("⚠ Silakan masukkan nama universitas.")
                else:
                    fitur = np.array([[academic, employer, citations, faculty_student]])
                    prediksi = model.predict(fitur)[0]

                    # Gabung data 2026 + prediksi
                    df_temp = df_2026.copy()
                    df_temp = pd.concat([df_temp, pd.DataFrame({
                        "institution": [nama_kampus],
                        "overall_score_2026": [prediksi]
                    })], ignore_index=True)

                    df_temp["rank_prediksi"] = df_temp["overall_score_2026"].rank(ascending=False, method="min").astype(int)

                    # Ambil rank prediksi
                    rank_pred = df_temp.loc[df_temp["institution"] == nama_kampus, "rank_prediksi"].values[0]

                    # Tampilkan hasil
                    st.success(f"🎯 Prediksi Overall Score: **{prediksi:.2f}**")
                    st.info(f"🏅 Perkiraan Peringkat: **#{rank_pred} dari {len(df_temp)} universitas**")
                    st.markdown(f"🎓 **{nama_kampus}** diprediksi memperoleh Overall Score **{prediksi:.2f}** dan berada di peringkat **{rank_pred}** di tahun 2026.")


# ------------------------
# MENU PERGESERAN
# ------------------------
elif menu == "Pergeseran Peringkat":
    st.title("📊 Pergeseran Peringkat Top 10")
    pilihan = st.selectbox("Tahun", [2023, 2025, 2026])
    if pilihan == 2026:
        df_top = df_2026.sort_values(by="overall_score_2026", ascending=False).head(10)
        col_score = "overall_score_2026"
    else:
        df_top = df_raw.sort_values(by=f"overall_score_{pilihan}", ascending=False).head(10)
        col_score = f"overall_score_{pilihan}"

    fig = px.bar(df_top, x=col_score, y="institution", orientation="h", color=col_score,
                 color_continuous_scale="blues", title=f"Top 10 Tahun {pilihan}")
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig, use_container_width=True)

# ------------------------
# MENU DATASET
# ------------------------
elif menu == "Tampilan Dataset":
    st.title("📄 Dataset Peringkat Universitas")
    years = st.sidebar.multiselect("Pilih Tahun", sorted(df_long["year"].unique()) + [2026], default=sorted(df_long["year"].unique()) + [2026])
    df_filtered = df_long[df_long["year"].isin([y for y in years if y != 2026])]
    if 2026 in years:
        df_2026_temp = df_2026.copy()
        df_2026_temp["year"] = 2026
        df_2026_temp = df_2026_temp.rename(columns={"overall_score_2026": "overall_score"})
        df_filtered = pd.concat([df_filtered, df_2026_temp[["institution", "year", "overall_score"]]], ignore_index=True)

    st.dataframe(df_filtered, use_container_width=True)
    csv = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Unduh CSV", csv, file_name="filtered_data.csv", mime="text/csv")
