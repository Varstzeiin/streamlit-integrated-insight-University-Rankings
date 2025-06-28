# app_unified.py (versi all-in-one dengan dropdown visualisasi prediksi)
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import joblib

# ------------------------
# Load data dan model
# ------------------------
df_raw = pd.read_csv("data_gabungan_lengkap.csv")
df_2026 = pd.read_csv("New_overscore_all.csv")

# Konversi data wide -> long untuk keperluan Dashboard dan Dataset
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
df_long["year"] = df_long["year"].str.extract(r"(\d{4})").astype(int)

# Load model prediksi
try:
    model = joblib.load("D:/Bootcamp-Offline-Bdg/Offline_Bootcamp[14]-Streamlit-Web_Rank_Univ/model_2026.pkl")
except:
    model = None

# ------------------------
# Sidebar: Menu Navigasi
# ------------------------
menu = st.sidebar.radio("Pilih Menu", ["Dashboard", "Prediksi", "Pergeseran Peringkat", "Tampilan Dataset"])

# ------------------------
# MENU 1: DASHBOARD
# ------------------------
if menu == "Dashboard":
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
    fig = px.bar(top10,
                 x="overall_score",
                 y="institution",
                 orientation="h",
                 color="overall_score",
                 color_continuous_scale="viridis",
                 title=f"10 Besar Universitas Tahun {year}")
    fig.update_layout(yaxis=dict(categoryorder='total ascending'))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Tren Rata-rata Overall Score (2018–2026)")
    avg_score_2026 = df_2026["overall_score_2026"].mean()
    avg_score = df_long.groupby("year")["overall_score"].mean().reset_index()
    avg_score = pd.concat([avg_score, pd.DataFrame({"year": [2026], "overall_score": [avg_score_2026]})], ignore_index=True)
    fig2 = px.line(avg_score, x="year", y="overall_score", markers=True,
                   title="Tren Rata-rata Overall Score per Tahun")
    st.plotly_chart(fig2, use_container_width=True)

# ------------------------
# MENU 2: PREDIKSI
# ------------------------
elif menu == "Prediksi":
    st.title("🧠 Prediksi Overall Score Universitas Baru")

    if model is None:
        st.error("Model belum tersedia. Pastikan model_2026.pkl berada di direktori.")
    else:
        with st.form("form_prediksi"):
            st.subheader("📥 Masukkan Nilai Fitur:")
            col1, col2 = st.columns(2)

            with col1:
                academic = st.number_input("Academic Reputation Score", 0.0, 100.0, step=0.1)
                employer = st.number_input("Employer Reputation Score", 0.0, 100.0, step=0.1)

            with col2:
                citations = st.number_input("Citations per Faculty", 0.0, 100.0, step=0.1)
                faculty_student = st.number_input("Faculty Student Score", 0.0, 100.0, step=0.1)

            submitted = st.form_submit_button("🔮 Prediksi Skor")

            if submitted:
                fitur = np.array([[academic, employer, citations, faculty_student]])
                prediksi = model.predict(fitur)[0]
                st.success(f"🎯 Prediksi Overall Score: **{prediksi:.2f}**")

# ------------------------
# MENU 3: PERGESERAN PERINGKAT
# ------------------------
elif menu == "Pergeseran Peringkat":
    st.title("📊 Pergeseran Peringkat Universitas Top 10")
    st.markdown("""
        Visualisasi berikut memperlihatkan perbandingan peringkat universitas terbaik dari tahun ke tahun.
        Pilih tahun untuk melihat Top 10 berdasarkan skor keseluruhan.
    """)

    pilihan = st.selectbox("Pilih Tahun", [2023, 2025, 2026])

    if pilihan == 2026:
        df_top = df_2026.sort_values(by="overall_score_2026", ascending=False).head(10)
        kolom_score = "overall_score_2026"
    elif pilihan == 2025:
        df_top = df_raw.sort_values(by="overall_score_2025", ascending=False).head(10)
        kolom_score = "overall_score_2025"
    elif pilihan == 2023:
        df_top = df_raw.sort_values(by="overall_score_2023", ascending=False).head(10)
        kolom_score = "overall_score_2023"

    fig_top = px.bar(df_top,
                     x=kolom_score,
                     y="institution",
                     orientation="h",
                     color=kolom_score,
                     color_continuous_scale="blues",
                     title=f"Top 10 Universitas Berdasarkan Overall Score Tahun {pilihan}")
    fig_top.update_layout(yaxis=dict(categoryorder='total ascending'))
    st.plotly_chart(fig_top, use_container_width=True)

# ------------------------
# MENU 4: TAMPILAN DATASET
# ------------------------
elif menu == "Tampilan Dataset":
    st.title("📄 Dataset Peringkat Universitas (2018–2026)")

    st.sidebar.header("Filter Data")
    years = st.sidebar.multiselect("Pilih Tahun", sorted(df_long["year"].unique()) + [2026], default=sorted(df_long["year"].unique()) + [2026])
    institutions = st.sidebar.multiselect("Pilih Universitas", sorted(df_raw["institution"].unique()))

    df_filtered = df_long[df_long["year"].isin([y for y in years if y != 2026])]
    if 2026 in years:
        df_2026_temp = df_2026.copy()
        df_2026_temp["year"] = 2026
        df_2026_temp = df_2026_temp.rename(columns={"overall_score_2026": "overall_score"})
        df_2026_temp = df_2026_temp[["institution", "year", "overall_score"]]
        df_filtered = pd.concat([df_filtered, df_2026_temp], ignore_index=True)

    if institutions:
        df_filtered = df_filtered[df_filtered["institution"].isin(institutions)]

    st.markdown(f"Menampilkan **{len(df_filtered)}** baris hasil filter.")
    st.dataframe(df_filtered, use_container_width=True)

    csv = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Unduh Data (CSV)", csv, file_name="filtered_university_data.csv", mime="text/csv")

# menjalankannya 
# streamlit run D:\Bootcamp-Offline-Bdg\Offline_Bootcamp[14]-Streamlit-Web_Rank_Univ\app.py
