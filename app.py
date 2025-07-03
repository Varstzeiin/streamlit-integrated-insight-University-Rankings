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
        st.sidebar.success(f"✅ Data tambahan dimuat: {df_uploaded.shape[0]} baris.")
        st.session_state["df_uploaded"] = df_uploaded
    except Exception as e:
        st.sidebar.error(f"⚠️ Gagal membaca file: {e}")

# ------------------------
# MENU HOME
# ------------------------
if menu == "Home":
    st.title("🌐 University Rankings & Performance Dashboard")
    st.image("Kampus.png", use_column_width=True, caption="Ilustrasi Kampus dan Peringkat")
    st.markdown("""
    ## Latar Belakang  
    Aplikasi ini untuk analisis & prediksi peringkat universitas global.  
    Data sumber dari QS dan dataset publik lainnya (2018–2026).

    ## Tujuan  
    - Prediksi peringkat 2025-2026  
    - Insight untuk strategi kolaborasi & talenta  

    ## Indikator  
    - Academic & Employer Reputation  
    - Citations per Faculty  
    - Faculty Student Ratio  
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
        st.error("⚠ Model belum tersedia.")
    else:
        with st.form("form_prediksi"):
            nama = st.text_input("Nama Universitas")
            col1, col2 = st.columns(2)
            academic = col1.number_input("Academic Reputation (0-100)", 0.0, 100.0, step=0.1)
            employer = col1.number_input("Employer Reputation (0-100)", 0.0, 100.0, step=0.1)
            citations = col2.number_input("Citations per Faculty (0-100)", 0.0, 100.0, step=0.1)
            faculty_student = col2.number_input("Faculty Student (0-100)", 0.0, 100.0, step=0.1)
            submit = st.form_submit_button("Prediksi")

        if submit:
            if nama.strip() == "":
                st.warning("Masukkan nama universitas!")
            else:
                fitur = np.array([[academic, employer, citations, faculty_student]])
                pred = model.predict(fitur)[0]
                df_temp = df_2026.copy()
                df_temp = pd.concat([df_temp, pd.DataFrame({"institution": [nama], "overall_score_2026": [pred]})], ignore_index=True)
                df_temp["rank_prediksi"] = df_temp["overall_score_2026"].rank(ascending=False, method="min").astype(int)
                rank = df_temp.loc[df_temp["institution"] == nama, "rank_prediksi"].values[0]
                st.success(f"🎯 Skor: {pred:.2f} | Ranking: #{rank}")

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
