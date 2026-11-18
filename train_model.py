import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "nhanes_cardio_sdoh.csv")
MODEL_PATH = os.path.join(BASE_DIR, "clinical_equity_model.joblib")

# Load CDC NHANES (2015-2020) merged clinical + SDOH dataset
df = pd.read_csv(DATASET_PATH)

FEATURE_COLS = [
    "Age",
    "Sex",
    "SysBP",
    "Hypertension",
    "TotalChol",
    "HighChol",
    "BMI",
    "Diabetes",
    "Smoker",
    "Stroke",
    "CKD",
    "FamilyHistory",
    "CancerHistory",
    "IncomeBracket",
    "HealthcareAccess",
    "FoodInsecurity",
    "DietQuality",
    "Education",
]

X = df[FEATURE_COLS]
y = df["HeartDiseaseorAttack"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Clinically grounded monotonic constraints (+1 increases risk, -1 decreases risk)
monotone_constraints = {
    "Age": 1,
    "Sex": 1,
    "SysBP": 1,
    "Hypertension": 1,
    "TotalChol": 1,
    "HighChol": 1,
    "BMI": 1,
    "Diabetes": 1,
    "Smoker": 1,
    "Stroke": 1,
    "CKD": 1,
    "FamilyHistory": 1,
    "CancerHistory": 1,
    "IncomeBracket": -1,
    "HealthcareAccess": 1,
    "FoodInsecurity": 1,
    "DietQuality": 1,
    "Education": -1,
}

model = xgb.XGBClassifier(
    n_estimators=250,
    max_depth=4,
    learning_rate=0.04,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    monotone_constraints=monotone_constraints,
    eval_metric="logloss",
    random_state=42,
)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
print(f"Trained on {len(X_train)} rows, evaluated on {len(X_test)} rows.")
print(f"Test ROC-AUC: {roc_auc_score(y_test, probs):.4f}")
print(f"Test Brier Score: {brier_score_loss(y_test, probs):.4f}")

joblib.dump(model, MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")
