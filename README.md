# ChurnGuard - Customer Churn Prediction Dashboard

**ChurnGuard** is an interactive web application for predicting customer churn using machine learning. Built as a final-year honours project, it helps businesses identify which customers are at risk of leaving.

## 💡 Features

- 🔍 Predict churn based on user input
- 📊 Live dashboard with real-time logs
- 🧠 Machine Learning model (Random Forest)
- 📄 Export predictions to CSV/PDF
- ☁️ Deployed using Streamlit Cloud

## 📁 Files

- `dashboard.py`: Main Streamlit app
- `model.pkl`: Trained ML model (uploaded manually)
- `scaler.pkl`: Preprocessing scaler (uploaded manually)
- `requirements.txt`: All dependencies
- `.streamlit/config.toml`: Theme config (forces light mode)

## 🚀 How to Run (Locally)

```bash
pip install -r requirements.txt
streamlit run dashboard.py
