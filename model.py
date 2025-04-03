import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# 1. Load and clean data
df = pd.read_csv("Churn_Modelling.csv")
df_cleaned = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
df_cleaned = pd.get_dummies(df_cleaned, columns=["Geography", "Gender"], drop_first=True)

# 2. Define features and target
X = df_cleaned.drop(columns=["Exited"])
y = df_cleaned["Exited"]

# 3. Balance classes using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# 5. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train models
log_model = LogisticRegression(random_state=42, max_iter=1000)
log_model.fit(X_train_scaled, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# 7. Evaluate models
def evaluate_model(y_true, y_pred, model_name):
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred)
    }

results = [
    evaluate_model(y_test, log_model.predict(X_test_scaled), "Logistic Regression"),
    evaluate_model(y_test, rf_model.predict(X_test_scaled), "Random Forest")
]

# Display results
results_df = pd.DataFrame(results)
print(results_df)

# 8. Save trained model and scaler
pickle.dump(rf_model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ Model and scaler saved as 'model.pkl' and 'scaler.pkl'")
