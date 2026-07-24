import os
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize Gemini Client
gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=gemini_api_key) if gemini_api_key else None

try:
    model = joblib.load('clinical_equity_model.joblib')
except FileNotFoundError:
    model = None
    print("Warning: Model file not found. Ensure you run train_model.py first.")


def map_form_to_features(data):
    """Translates frontend JSON to the exact BRFSS numeric columns."""
    prev_diseases = data.get('prev_diseases', [])
    sysbp = float(data.get('sysbp', 120))
    chol = float(data.get('chol', 200))
    age = float(data.get('age', 55))

    income_map = {'<25k': 1, '25k-50k': 3, '50k-75k': 5, '75k-100k': 7, '>100k': 8}
    brfss_age_cat = min(max(int((age - 18) / 5) + 1, 1), 13)

    feature_dict = {
        'HighBP': 1 if (sysbp >= 130 or 'hypertension' in prev_diseases) else 0,
        'HighChol': 1 if chol >= 200 else 0,
        'CholCheck': 1,
        'BMI': 27.0,
        'Smoker': 1 if data.get('smoker') == 'current' else 0,
        'Stroke': 1 if 'stroke' in prev_diseases else 0,
        'PhysActivity': 1,
        'Fruits': 1 if data.get('diet') in ['mediterranean', 'dash', 'plant_based'] else 0,
        'Veggies': 1 if data.get('diet') in ['mediterranean', 'dash', 'plant_based'] else 0,
        'HvyAlcoholConsump': 1 if data.get('drinking') == 'heavy' else 0,
        'AnyHealthcare': 1 if data.get('healthcare') == 'fully_insured' else 0,
        'NoDocbcCost': 1 if data.get('healthcare') == 'uninsured' else 0,
        'GenHlth': 3,
        'MentHlth': 0,
        'PhysHlth': 0,
        'DiffWalk': 1 if 'pvd' in prev_diseases else 0,
        'Sex': 1 if data.get('gender') == 'male' else 0,
        'Age': brfss_age_cat,
        'Education': 5,
        'Income': income_map.get(data.get('income'), 5)
    }

    return pd.DataFrame([feature_dict]).astype(float)


def generate_recommendations(risk):
    if risk < 5.0:
        return {"tier": "Low Risk", "action": "Maintain healthy lifestyle behaviors.",
                "clinical": "Statin therapy generally not indicated."}
    elif 5.0 <= risk < 7.5:
        return {"tier": "Borderline Risk", "action": "Intensify lifestyle modifications.",
                "clinical": "Consider a CAC scan. Discuss moderate-intensity statins."}
    elif 7.5 <= risk < 20.0:
        return {"tier": "Intermediate Risk", "action": "Immediate lifestyle modifications required.",
                "clinical": "Moderate to high-intensity statin therapy strongly recommended."}
    else:
        return {"tier": "High Risk", "action": "Urgent risk factor modification.",
                "clinical": "High-intensity statin therapy indicated. Urgent cardiology referral."}


def generate_ai_protocol(data, base_risk, adjusted_risk):
    """Uses Gemini to verify risks and generate a tailored clinical summary."""
    if not gemini_client:
        return "AI Protocol unavailable. Please configure your GEMINI_API_KEY in the .env file."

    prompt = f"""
    You are an AI Clinical Navigator assessing a patient's cardiovascular risk.

    - Standard Biological Risk: {base_risk}%
    - Context-Adjusted Risk: {adjusted_risk}%
    - SDOH Inputs: Income: {data.get('income')}, Diet: {data.get('diet')}, Healthcare Access: {data.get('healthcare')}, Food Access: {data.get('food_access')}.

    First, verify if the results make sense physically (e.g., risk cannot be > 100%). If impossible, return "Error: Risk parameters out of bounds."

    Then, write a concise, professional clinical summary (2-3 sentences max):
    - If the Adjusted Risk is HIGHER than the Standard Risk: Explain which specific structural barriers (from the inputs) are driving the gap and suggest one actionable clinic-level step to help (e.g., social worker referral).
    - If the risks are the SAME (0% gap): Acknowledge that the patient's socioeconomic environment is highly protective (no SDOH penalties) and advise that treatment should focus purely on their biological/lifestyle risk factors.

    Do not use bullet points or bold text. Keep the tone medical and empathetic.
    """
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ["429", "503", "demand", "quota", "exhausted"]):
            return "The AI model is currently experiencing high demand. Please try generating the protocol again in a few moments."
        return f"Error generating protocol: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')


def validate_clinical_inputs(data):
    """
    Checks physiological inputs against hard boundaries to prevent
    impossible data from breaking the ML model or math functions.
    Returns a list of error strings if any are found.
    """
    errors = []

    try:
        age = float(data.get('age', 55))
        sysbp = float(data.get('sysbp', 120))
        chol = float(data.get('chol', 200))

        if age < 18 or age > 110:
            errors.append("Age must be between 18 and 110.")

        if sysbp < 70 or sysbp > 350:
            errors.append(f"Systolic BP of {sysbp} is outside valid physiological bounds (70-350 mmHg). Please verify.")

        if chol < 80 or chol > 600:
            errors.append(f"Total Cholesterol of {chol} is outside valid bounds (80-600 mg/dL). Please verify.")

    except ValueError:
        errors.append("Vitals must be valid numeric values.")

    return errors

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.json

    # 1. Strict Boundary Validation (Replaces the basic age check)
    validation_errors = validate_clinical_inputs(data)
    if validation_errors:
        # Join multiple errors into a single string to send to the UI
        error_message = " | ".join(validation_errors)
        return jsonify({'error': error_message}), 400

    # 2. Map inputs for XGBoost
    df = map_form_to_features(data)

    # ... (The rest of your existing ML and hazard calculation logic remains exactly the same) ...

    # 2. Calculate ML Baseline
    df_baseline = df.copy()
    df_baseline['Income'] = 8.0
    df_baseline['AnyHealthcare'] = 1.0
    df_baseline['NoDocbcCost'] = 0.0
    df_baseline['Fruits'] = 1.0
    df_baseline['Veggies'] = 1.0

    base_risk = 0.0
    if model:
        base_risk = float(model.predict_proba(df_baseline)[0][1] * 100)

    # ---------------------------------------------------------
    # NEW: CLINICAL REDLINE OVERRIDE
    # Exponentially scales risk if vitals enter crisis territory
    # ---------------------------------------------------------
    sysbp = float(data.get('sysbp', 120))
    chol = float(data.get('chol', 200))

    # Hypertensive crisis threshold (AHA) is > 180
    if sysbp > 180:
        bp_scalar = (sysbp / 140.0) ** 3  # Cubed penalty for extreme BP
        base_risk = base_risk * bp_scalar

    # Extreme hyperlipidemia threshold
    if chol > 300:
        chol_scalar = (chol / 200.0) ** 2
        base_risk = base_risk * chol_scalar

    # Cap biological baseline at 99.0% before SDOH application
    base_risk = min(base_risk, 99.0)
    # ---------------------------------------------------------

    # 3. Apply Deterministic Hazard Penalties for SDOH
    multiplier = 1.0
    income = data.get('income', '>100k')
    if income == '<25k':
        multiplier += 0.35
    elif income == '25k-50k':
        multiplier += 0.20

    healthcare = data.get('healthcare', 'fully_insured')
    if healthcare == 'uninsured':
        multiplier += 0.30
    elif healthcare == 'underinsured':
        multiplier += 0.15

    if data.get('food_access') == 'desert': multiplier += 0.15
    if data.get('diet') == 'ultra_processed': multiplier += 0.20

    adjusted_risk = min(base_risk * multiplier, 99.9)

    # 4. Generate AI Protocol
    ai_protocol = generate_ai_protocol(data, round(base_risk, 1), round(adjusted_risk, 1))

    return jsonify({
        'base_risk': round(base_risk, 1),
        'adjusted_risk': round(adjusted_risk, 1),
        'risk_gap': round(adjusted_risk - base_risk, 1),
        'recommendations': generate_recommendations(adjusted_risk),
        'ai_protocol': ai_protocol
    })


@app.route('/chat', methods=['POST'])
def chat():
    if not gemini_client:
        return jsonify({"response": "Gemini API key is missing."}), 500

    data = request.json
    user_message = data.get("message", "")
    system_context = "You are an AI Clinical Navigator embedded in a Cardiovascular Risk Calculator. Keep responses concise and medical."
    try:
        response = gemini_client.models.generate_content(model="gemini-2.5-flash",
                                                         contents=[system_context, user_message])
        return jsonify({"response": response.text})
    except Exception as e:
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ["429", "503", "demand", "quota", "exhausted"]):
            return jsonify(
                {"response": "I am currently experiencing high demand. Please try asking again in a minute."})
        return jsonify({"response": f"Error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)