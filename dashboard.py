import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# === CONFIG ===
st.set_page_config(page_title="ChurnGuard", layout="wide")

# === COLORS ===
PRIMARY = "#00BFFF"      # Sky Blue
ACCENT = "#FFFACD"       # Soft Yellow
TEXT_DARK = "#1e1e1e"
SUBTEXT = "#555555"
WHITE = "#ffffff"

# === LOAD MODEL & SCALER ===
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# === SIDEBAR ===
st.sidebar.markdown(f"""
    <div style="background-color:{PRIMARY};padding:20px;border-radius:10px;">
        <h1 style="color:{TEXT_DARK};text-align:center;">🛡️ ChurnGuard</h1>
    </div>
""", unsafe_allow_html=True)

nav = st.sidebar.radio("Navigate", ["🏠 Home", "🔍 Predict", "📁 Logs", "ℹ️ About"])

# === HEADER ===
st.markdown(f"""
    <div style="background-color:{ACCENT};padding:20px;border-radius:12px;">
        <h1 style="color:{TEXT_DARK};text-align:center;">ChurnGuard Dashboard</h1>
        <p style="color:{SUBTEXT};text-align:center;">Predict and monitor customer churn with real-time insights</p>
    </div>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# === HOME ===
if nav == "🏠 Home":
    st.subheader("📊 Dashboard Overview")

    try:
        df_logs = pd.read_csv("predictions.csv")

        # 🛠 Fix Timestamp Conversion
        df_logs['Timestamp'] = pd.to_datetime(df_logs['Timestamp'], errors='coerce')
        df_logs = df_logs.dropna(subset=['Timestamp'])

        total = len(df_logs)
        churn = len(df_logs[df_logs["Prediction"] == "Churn"])
        no_churn = total - churn
        churn_rate = (churn / total) * 100 if total > 0 else 0

        # === STAT CARDS ===
        col1, col2, col3 = st.columns(3)
        col1.metric("📦 Total Predictions", f"{total}")
        col2.metric("❌ Churned", f"{churn} ({churn_rate:.1f}%)")
        col3.metric("✅ Retained", f"{no_churn} ({100 - churn_rate:.1f}%)")

        # === PIE CHART ===
        st.markdown("### 🔄 Churn Distribution")
        pie = px.pie(df_logs, names="Prediction", color_discrete_sequence=[PRIMARY, "#FFA07A"])
        st.plotly_chart(pie, use_container_width=True)

        # === LINE CHART ===
        st.markdown("### 📈 Predictions Over Time")
        trend = df_logs.groupby(df_logs['Timestamp'].dt.date)['Prediction'].value_counts().unstack().fillna(0)
        line = px.line(trend, markers=True, color_discrete_sequence=[PRIMARY, "#FFA500"])
        st.plotly_chart(line, use_container_width=True)

    except FileNotFoundError:
        st.info("No predictions available yet. Go to 🔍 Predict to get started.")

# === PREDICT PAGE ===
elif nav == "🔍 Predict":
    st.subheader("🔍 Make a Prediction")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        credit_score = col1.number_input("Credit Score", 300, 850, 650)
        age = col2.slider("Age", 18, 100, 35)
        tenure = col3.selectbox("Tenure", list(range(11)), index=3)

        col4, col5, col6 = st.columns(3)
        balance = col4.number_input("Balance", 0.0, 250000.0, 50000.0)
        num_products = col5.selectbox("Number of Products", [1, 2, 3, 4], index=1)
        estimated_salary = col6.number_input("Estimated Salary", 0.0, 200000.0, 60000.0)

        col7, col8, col9 = st.columns(3)
        has_cr_card = col7.radio("Has Credit Card?", [1, 0])
        is_active = col8.radio("Active Member?", [1, 0])
        gender = col9.selectbox("Gender", ["Male", "Female"])

        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

        submit = st.form_submit_button("🚀 Predict")

    if submit:
        # Encode
        geo_germany = 1 if geography == "Germany" else 0
        geo_spain = 1 if geography == "Spain" else 0
        gender_male = 1 if gender == "Male" else 0

        input_df = pd.DataFrame([{
            "CreditScore": credit_score,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": has_cr_card,
            "IsActiveMember": is_active,
            "EstimatedSalary": estimated_salary,
            "Geography_Germany": geo_germany,
            "Geography_Spain": geo_spain,
            "Gender_Male": gender_male
        }])

        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1] * 100

        st.markdown("---")
        if prediction == 1:
            st.error(f"❌ This customer is likely to CHURN.\n\n💣 Probability: {probability:.2f}%")
        else:
            st.success(f"✅ This customer is likely to STAY.\n\n🛡️ Probability: {probability:.2f}%")

        # Save to CSV
        input_df["Prediction"] = "Churn" if prediction == 1 else "No Churn"
        input_df["Churn_Probability"] = round(probability, 2)
        input_df["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            input_df.to_csv("predictions.csv", mode='a', header=not pd.io.common.file_exists("predictions.csv"), index=False)
        except Exception as e:
            st.warning(f"Could not save prediction: {e}")

# === LOG PAGE ===
elif nav == "📁 Logs":
    st.subheader("📁 Prediction Logs")
    try:
        df_logs = pd.read_csv("predictions.csv")
        st.dataframe(df_logs, use_container_width=True)

        csv = df_logs.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Logs", csv, "predictions.csv", "text/csv")
    except FileNotFoundError:
        st.info("No logs found.")

# === ABOUT ===
elif nav == "ℹ️ About":
    st.subheader("ℹ️ About ChurnGuard")
    st.markdown(f"""
        <div style="background-color:{ACCENT};padding:20px;border-radius:10px;">
        <p style="color:{TEXT_DARK};">
        ChurnGuard is a machine learning-powered dashboard built for predicting and analyzing customer churn.
        <br><br>
        📊 Visualize trends<br>
        🧠 Predict churn risk<br>
        💾 Save and analyze customer data over time<br><br>
        Developed By Shahriyar.
        </p>
        </div>
    """, unsafe_allow_html=True)
