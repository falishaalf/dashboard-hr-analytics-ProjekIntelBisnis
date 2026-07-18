"""
=====================================================================
DASHBOARD INTELIJEN BISNIS - HR ANALYTICS (VERSI SIMPLE)
=====================================================================
Isi dashboard:
1. Dashboard    -> KPI + grafik ringkas
2. Visualisasi  -> beberapa grafik eksplorasi data
3. Clustering   -> K-Means, segmentasi karyawan
4. Klasifikasi  -> Decision Tree, prediksi Attrition (resign/tidak)
5. Regresi      -> Linear Regression, prediksi gaji (MonthlyIncome)

Cara menjalankan:
    streamlit run streamlit_hr_dashboard.py
=====================================================================
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score

st.set_page_config(page_title="HR Analytics Dashboard", layout="wide")

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
df = pd.read_csv("HR_Analytics_clean.csv")

# ---------------------------------------------------------------------------
# 2. SIDEBAR: MENU + FILTER
# ---------------------------------------------------------------------------
st.sidebar.title("HR Analytics")
menu = st.sidebar.radio(
    "Pilih Menu",
    ["Dashboard", "Visualisasi", "Clustering", "Klasifikasi (Attrition)", "Regresi (Gaji)"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filter Data")
pilih_dept = st.sidebar.multiselect(
    "Department", df["Department"].unique(), default=list(df["Department"].unique())
)
df_filter = df[df["Department"].isin(pilih_dept)]


# ---------------------------------------------------------------------------
# HALAMAN 1: DASHBOARD
# ---------------------------------------------------------------------------
if menu == "Dashboard":
    st.title("Dashboard HR Analytics")

    total_karyawan = len(df_filter)
    total_resign = (df_filter["Attrition"] == "Yes").sum()
    attrition_rate = total_resign / total_karyawan * 100
    rata_gaji = df_filter["MonthlyIncome"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Karyawan", total_karyawan)
    col2.metric("Karyawan Resign", total_resign)
    col3.metric("Attrition Rate", f"{attrition_rate:.1f}%")
    col4.metric("Rata-rata Gaji", f"${rata_gaji:,.0f}")

    st.subheader("Attrition per Department")
    grafik = df_filter.groupby(["Department", "Attrition"]).size().reset_index(name="Jumlah")
    fig = px.bar(grafik, x="Department", y="Jumlah", color="Attrition", barmode="group")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Tabel Data")
    st.dataframe(df_filter, width="stretch")


# ---------------------------------------------------------------------------
# HALAMAN 2: VISUALISASI
# ---------------------------------------------------------------------------
elif menu == "Visualisasi":
    st.title("Visualisasi Data")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribusi Usia Karyawan")
        fig = px.histogram(df_filter, x="Age", color="Attrition", nbins=20)
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("Gaji per Job Role")
        fig = px.box(df_filter, x="JobRole", y="MonthlyIncome")
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Work-Life Balance vs Attrition")
    wlb = df_filter.groupby(["WorkLifeBalance", "Attrition"]).size().reset_index(name="Jumlah")
    fig = px.bar(wlb, x="WorkLifeBalance", y="Jumlah", color="Attrition", barmode="group")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Proporsi Gender")
    gen = df_filter["Gender"].value_counts().reset_index()
    gen.columns = ["Gender", "Jumlah"]
    fig = px.pie(gen, names="Gender", values="Jumlah")
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# HALAMAN 3: CLUSTERING (K-MEANS) -> unsupervised learning
# ---------------------------------------------------------------------------
elif menu == "Clustering":
    st.title("Segmentasi Karyawan (K-Means Clustering)")
    st.write("Mengelompokkan karyawan berdasarkan kemiripan karakteristik tanpa label, "
             "menggunakan kolom **Age, MonthlyIncome, YearsAtCompany**.")

    jumlah_cluster = st.slider("Jumlah Cluster (k)", 2, 6, 3)

    fitur = ["Age", "MonthlyIncome", "YearsAtCompany"]
    X = df_filter[fitur]
    X_scaled = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=jumlah_cluster, random_state=42, n_init=10)
    df_filter = df_filter.copy()
    df_filter["Cluster"] = kmeans.fit_predict(X_scaled).astype(str)

    st.subheader("Visualisasi Cluster")
    fig = px.scatter(df_filter, x="Age", y="MonthlyIncome", color="Cluster",
                      hover_data=["YearsAtCompany", "Attrition"])
    st.plotly_chart(fig, width="stretch")

    st.subheader("Profil Rata-rata Tiap Cluster")
    profil = df_filter.groupby("Cluster")[fitur].mean().round(1)
    profil["Jumlah Karyawan"] = df_filter["Cluster"].value_counts()
    st.dataframe(profil, width="stretch")

    st.info("Cluster dengan rata-rata gaji rendah & masa kerja pendek bisa jadi "
            "target program retensi karyawan.")


# ---------------------------------------------------------------------------
# HALAMAN 4: KLASIFIKASI (DECISION TREE) -> supervised learning
# ---------------------------------------------------------------------------
elif menu == "Klasifikasi (Attrition)":
    st.title("Prediksi Attrition Karyawan (Decision Tree)")
    st.write("Memprediksi apakah karyawan berpotensi **resign** atau **bertahan** "
             "berdasarkan data historis.")

    fitur_numerik = ["Age", "MonthlyIncome", "DistanceFromHome", "JobSatisfaction",
                      "WorkLifeBalance", "YearsAtCompany", "TotalWorkingYears"]
    fitur_kategorikal = ["OverTime", "Department"]

    data_model = df[fitur_numerik + fitur_kategorikal + ["Attrition"]].dropna()
    X = pd.get_dummies(data_model[fitur_numerik + fitur_kategorikal],
                        columns=fitur_kategorikal, drop_first=True)
    y = data_model["Attrition"].map({"Yes": 1, "No": 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    akurasi = accuracy_score(y_test, y_pred)

    st.metric("Akurasi Model", f"{akurasi*100:.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                         x=["Tidak Resign", "Resign"], y=["Tidak Resign", "Resign"],
                         labels=dict(x="Prediksi", y="Aktual"))
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("Feature Importance")
        fi = pd.DataFrame({"Fitur": X.columns, "Importance": model.feature_importances_})
        fi = fi.sort_values("Importance").tail(8)
        fig = px.bar(fi, x="Importance", y="Fitur", orientation="h")
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("Coba Prediksi Karyawan Baru")
    with st.form("form_prediksi"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", value=30)
            income = st.number_input("MonthlyIncome", value=5000)
        with c2:
            distance = st.number_input("DistanceFromHome", value=5)
            jobsat = st.slider("JobSatisfaction", 1, 4, 3)
        with c3:
            wlb = st.slider("WorkLifeBalance", 1, 4, 3)
            years = st.number_input("YearsAtCompany", value=3)
            total_years = st.number_input("TotalWorkingYears", value=5)
        overtime = st.selectbox("OverTime", df["OverTime"].unique())
        dept = st.selectbox("Department", df["Department"].unique())
        submit = st.form_submit_button("Prediksi")

    if submit:
        input_df = pd.DataFrame([{
            "Age": age, "MonthlyIncome": income, "DistanceFromHome": distance,
            "JobSatisfaction": jobsat, "WorkLifeBalance": wlb, "YearsAtCompany": years,
            "TotalWorkingYears": total_years, "OverTime": overtime, "Department": dept
        }])
        input_encoded = pd.get_dummies(input_df, columns=fitur_kategorikal, drop_first=True)
        input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)
        pred = model.predict(input_encoded)[0]
        prob = model.predict_proba(input_encoded)[0][1]

        if pred == 1:
            st.error(f"Karyawan diprediksi BERPOTENSI RESIGN (probabilitas {prob*100:.1f}%)")
        else:
            st.success(f"Karyawan diprediksi BERTAHAN (probabilitas resign {prob*100:.1f}%)")


# ---------------------------------------------------------------------------
# HALAMAN 5: REGRESI (LINEAR REGRESSION) -> supervised learning
# ---------------------------------------------------------------------------
elif menu == "Regresi (Gaji)":
    st.title("Prediksi Gaji Bulanan (Linear Regression)")
    st.write("Memperkirakan **MonthlyIncome** berdasarkan level jabatan dan pengalaman kerja.")

    fitur = ["JobLevel", "TotalWorkingYears", "YearsAtCompany", "Education"]
    X = df[fitur]
    y = df["MonthlyIncome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    st.metric("R² Score", f"{r2:.3f}")

    st.subheader("Aktual vs Prediksi")
    hasil = pd.DataFrame({"Aktual": y_test, "Prediksi": y_pred})
    fig = px.scatter(hasil, x="Aktual", y="Prediksi")
    st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("Coba Prediksi Gaji Karyawan Baru")
    with st.form("form_gaji"):
        c1, c2 = st.columns(2)
        with c1:
            joblevel = st.slider("JobLevel", 1, 5, 2)
            totalyears = st.number_input("TotalWorkingYears", value=5)
        with c2:
            years = st.number_input("YearsAtCompany", value=3)
            edu = st.slider("Education", 1, 5, 3)
        submit2 = st.form_submit_button("Prediksi Gaji")

    if submit2:
        input_df = pd.DataFrame([{
            "JobLevel": joblevel, "TotalWorkingYears": totalyears,
            "YearsAtCompany": years, "Education": edu
        }])
        hasil_prediksi = model.predict(input_df)[0]
        st.success(f"Prediksi Monthly Income: ${hasil_prediksi:,.0f}")
