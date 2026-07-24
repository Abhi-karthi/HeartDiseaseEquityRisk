import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib

# 1. Load the 50/50 split dataset
df = pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')

# 2. Define your target (y) and features (X)
# We are dropping HeartDiseaseorAttack because it's our target.
# We can also drop Diabetes_binary if we purely want to focus on cardiovascular risk from vitals/SDOH.
y = df['HeartDiseaseorAttack']
X = df.drop(['HeartDiseaseorAttack', 'Diabetes_binary'], axis=1)

# 3. Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the XGBoost Model
# XGBoost is great here because it natively handles the 1-8 ordinal scales used for Income
model = xgb.XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)

# 5. Save the model to use in your Flask app
joblib.dump(model, 'clinical_equity_model.joblib')
