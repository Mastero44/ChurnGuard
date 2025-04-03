from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load model and scaler with error handling
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except Exception as e:
    print(f"❌ Error loading model or scaler: {e}")
    model, scaler = None, None

@app.route('/')
def home():
    return render_template('form.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return render_template('form.html', prediction="Model not loaded. Please check your files.")

    try:
        # Extract and validate form data
        data = request.form
        credit_score = int(data.get('CreditScore'))
        age = int(data.get('Age'))
        tenure = int(data.get('Tenure'))
        balance = float(data.get('Balance'))
        num_products = int(data.get('NumOfProducts'))
        has_cr_card = int(data.get('HasCrCard'))
        is_active_member = int(data.get('IsActiveMember'))
        estimated_salary = float(data.get('EstimatedSalary'))
        geography = data.get('Geography')
        gender = data.get('Gender')

        # Manual encoding
        geo_germany = 1 if geography == 'Germany' else 0
        geo_spain = 1 if geography == 'Spain' else 0
        gender_male = 1 if gender == 'Male' else 0

        # Create DataFrame for input
        input_df = pd.DataFrame([{
            "CreditScore": credit_score,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": has_cr_card,
            "IsActiveMember": is_active_member,
            "EstimatedSalary": estimated_salary,
            "Geography_Germany": geo_germany,
            "Geography_Spain": geo_spain,
            "Gender_Male": gender_male
        }])

        # Scale input and predict
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1] * 100

        # Message for display
        result = "❌ Customer is likely to churn." if prediction == 1 else "✅ Customer is likely to stay."
        probability = round(probability, 2)

        return render_template('form.html', prediction=result, probability=probability)

    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return render_template('form.html', prediction="⚠️ Something went wrong. Please check your inputs.")

# Optional: JSON API for integration/testing
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()

        df = pd.DataFrame([{
            "CreditScore": int(data['CreditScore']),
            "Age": int(data['Age']),
            "Tenure": int(data['Tenure']),
            "Balance": float(data['Balance']),
            "NumOfProducts": int(data['NumOfProducts']),
            "HasCrCard": int(data['HasCrCard']),
            "IsActiveMember": int(data['IsActiveMember']),
            "EstimatedSalary": float(data['EstimatedSalary']),
            "Geography_Germany": 1 if data['Geography'] == 'Germany' else 0,
            "Geography_Spain": 1 if data['Geography'] == 'Spain' else 0,
            "Gender_Male": 1 if data['Gender'] == 'Male' else 0
        }])

        df_scaled = scaler.transform(df)
        prediction = model.predict(df_scaled)[0]
        probability = model.predict_proba(df_scaled)[0][1] * 100

        return jsonify({
            "prediction": "Churn" if prediction == 1 else "No Churn",
            "probability": round(probability, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
