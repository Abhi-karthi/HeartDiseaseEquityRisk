import math
import os
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = Flask(__name__)

# Gemini for in-app clinical summary and chat window
gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=gemini_api_key) if gemini_api_key else None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "clinical_equity_model.joblib")

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None


def map_form_to_features(data):
    """Translates frontend JSON to the 18 CDC NHANES clinical + SDOH model features."""
    prev_diseases = data.get("prev_diseases", [])
    age = float(data.get("age", 55))
    sysbp = float(data.get("sysbp", 120))
    chol = float(data.get("chol", 200))

    income_map = {
        "<25k": 1.0,
        "25k-50k": 2.0,
        "50k-75k": 3.0,
        "75k-100k": 4.0,
        ">100k": 5.0,
    }
    healthcare_map = {
        "fully_insured": 0.0,
        "underinsured": 1.0,
        "uninsured": 2.0,
    }
    smoker_map = {
        "never": 0.0,
        "former": 1.0,
        "current": 2.0,
    }
    diet_map = {
        "mediterranean": 1.0,
        "dash": 2.0,
        "plant_based": 2.0,
        "standard": 3.0,
        "ultra_processed": 5.0,
    }

    income_bracket = income_map.get(data.get("income"), 3.0)
    # Approximate NHANES education tier from socioeconomic bracket when not explicitly asked
    education_tier = min(max(income_bracket, 2.0), 5.0)

    feature_dict = {
        "Age": age,
        "Sex": 1.0 if data.get("gender") == "male" else 0.0,
        "SysBP": sysbp,
        "Hypertension": 1.0 if (sysbp >= 130 or "hypertension" in prev_diseases) else 0.0,
        "TotalChol": chol,
        "HighChol": 1.0 if chol >= 200 else 0.0,
        "BMI": 27.0,
        "Diabetes": 1.0 if data.get("diabetes") == "yes" else 0.0,
        "Smoker": smoker_map.get(data.get("smoker"), 0.0),
        "Stroke": 1.0 if "stroke" in prev_diseases else 0.0,
        "CKD": 1.0 if "ckd" in prev_diseases else 0.0,
        "FamilyHistory": 1.0 if data.get("family_history") == "yes" else 0.0,
        "CancerHistory": 1.0 if data.get("cancer_history") == "cardiotoxic" else 0.0,
        "IncomeBracket": income_bracket,
        "HealthcareAccess": healthcare_map.get(data.get("healthcare"), 0.0),
        "FoodInsecurity": 1.0 if data.get("food_access") == "desert" else 0.0,
        "DietQuality": diet_map.get(data.get("diet"), 3.0),
        "Education": education_tier,
    }

    return pd.DataFrame([feature_dict]).astype(float)


def calculate_clinical_formulas(data):
    """
    Calculates 10-Year ASCVD Risk using the published ACC/AHA Pooled Cohort Equations
    (Goff et al., Circulation 2014) combined with 2018/2019 ACC/AHA Guideline
    Risk-Enhancing Factors (Arnett et al., Circulation 2019) and AHA Scientific Statement
    SDOH Hazard Ratios (Schultz et al., Circulation 2018; Khan et al., Circulation 2024)
    via Cox proportional hazards survival transformation: P_adj = 1 - (1 - P_base)^HR.
    """
    age = float(data.get("age", 55))
    gender = data.get("gender", "male")
    ethnicity = data.get("ethnicity", "white")
    sysbp = float(data.get("sysbp", 120))
    chol = float(data.get("chol", 200))
    prev_diseases = data.get("prev_diseases", [])

    treated_bp = "hypertension" in prev_diseases
    smoker_current = data.get("smoker") == "current"
    diabetes = data.get("diabetes") == "yes"

    # 1. Core ACC/AHA Pooled Cohort Equations (Goff et al., 2014)
    # Evaluate polynomial at [40, 79] and apply smooth proportional hazards age scaling outside 40-79
    pce_age = min(max(age, 40.0), 79.0)
    ln_age = math.log(pce_age)
    ln_chol = math.log(max(chol, 80.0))
    hdl = 48.0 if gender == "male" else 58.0  # NHANES sex-specific median HDL-C (mg/dL)
    ln_hdl = math.log(hdl)
    ln_sbp = math.log(max(sysbp, 70.0))
    smk = 1.0 if smoker_current else 0.0
    dm = 1.0 if diabetes else 0.0

    is_black = ethnicity == "black"
    if gender == "female" and is_black:
        sbp_term = (
            (29.291 * ln_sbp - 6.432 * ln_age * ln_sbp)
            if treated_bp
            else (27.820 * ln_sbp - 6.087 * ln_age * ln_sbp)
        )
        ind_sum = (
            17.114 * ln_age
            + 0.940 * ln_chol
            - 18.920 * ln_hdl
            + 4.475 * ln_age * ln_hdl
            + sbp_term
            + 0.691 * smk
            + 0.874 * dm
        )
        s0, mean_xb = 0.9533, 86.61
    elif gender == "female" and not is_black:
        sbp_term = (2.019 * ln_sbp) if treated_bp else (1.957 * ln_sbp)
        ind_sum = (
            -29.799 * ln_age
            + 4.884 * (ln_age ** 2)
            + 13.540 * ln_chol
            - 3.114 * ln_age * ln_chol
            - 13.578 * ln_hdl
            + 3.149 * ln_age * ln_hdl
            + sbp_term
            + 7.574 * smk
            - 1.665 * ln_age * smk
            + 0.661 * dm
        )
        s0, mean_xb = 0.9665, -29.18
    elif gender == "male" and is_black:
        sbp_term = (1.916 * ln_sbp) if treated_bp else (1.809 * ln_sbp)
        ind_sum = (
            2.469 * ln_age
            + 0.302 * ln_chol
            - 0.307 * ln_hdl
            + sbp_term
            + 0.549 * smk
            + 0.645 * dm
        )
        s0, mean_xb = 0.8954, 19.54
    else:
        sbp_term = (1.797 * ln_sbp) if treated_bp else (1.764 * ln_sbp)
        ind_sum = (
            12.344 * ln_age
            + 11.853 * ln_chol
            - 2.664 * ln_age * ln_chol
            - 7.990 * ln_hdl
            + 1.769 * ln_age * ln_hdl
            + sbp_term
            + 7.837 * smk
            - 1.795 * ln_age * smk
            + 0.658 * dm
        )
        s0, mean_xb = 0.9144, 61.18

    pce_hazard = math.exp(ind_sum - mean_xb)
    if age < 40.0:
        pce_hazard *= (age / 40.0) ** 2.5
    elif age > 79.0:
        pce_hazard *= 1.06 ** (age - 79.0)

    # 2. 2018/2019 ACC/AHA Guideline Clinical Risk-Enhancing Hazard Ratios
    clinical_hr = 1.0
    if "stroke" in prev_diseases:
        clinical_hr *= 2.00  # Secondary prevention / prior cerebrovascular event
    if "pvd" in prev_diseases:
        clinical_hr *= 1.60  # Peripheral arterial disease
    if "ckd" in prev_diseases:
        clinical_hr *= 1.50  # Chronic kidney disease (CKD Prognosis Consortium / PREVENT)
    if data.get("family_history") == "yes":
        clinical_hr *= 1.40  # Premature family history of ASCVD (Lloyd-Jones et al., JAMA)
    if data.get("smoker") == "former":
        clinical_hr *= 1.25  # Former smoker residual hazard (Duncan et al., JAMA 2019)
    if data.get("cancer_history") == "cardiotoxic":
        clinical_hr *= 1.35  # Prior cardiotoxic chemotherapy/thoracic radiation
    if data.get("drugs") == "stimulants":
        clinical_hr *= 1.45  # Cocaine/stimulant-associated cardiovascular hazard (AHA 2010)
    if ethnicity == "indigenous":
        clinical_hr *= 1.15  # AHA Scientific Statement on AI/AN cardiovascular disparity

    # Continuous vascular overload above standard cohort calibration range (Lewington et al., Lancet 2002)
    if sysbp > 180.0:
        clinical_hr *= 2.0 ** ((sysbp - 180.0) / 40.0)
    if chol > 280.0:
        clinical_hr *= 1.35 ** ((chol - 280.0) / 80.0)

    base_prob = 1.0 - (s0 ** (pce_hazard * clinical_hr))
    base_prob = min(max(base_prob, 0.001), 0.990)

    # 3. Published SDOH Hazard Ratios (Schultz et al., Circulation 2018; AHA PREVENT SDI)
    sdoh_hr = 1.0
    income_hr_map = {
        "<25k": 1.40,
        "25k-50k": 1.25,
        "50k-75k": 1.12,
        "75k-100k": 1.05,
        ">100k": 1.00,
    }
    sdoh_hr *= income_hr_map.get(data.get("income", ">100k"), 1.00)

    healthcare_hr_map = {
        "uninsured": 1.30,
        "underinsured": 1.15,
        "fully_insured": 1.00,
    }
    sdoh_hr *= healthcare_hr_map.get(data.get("healthcare", "fully_insured"), 1.00)

    if data.get("food_access") == "desert":
        sdoh_hr *= 1.20  # USDA food desert / food insecurity hazard ratio (Sun et al., JAHA 2020)

    diet_hr_map = {
        "ultra_processed": 1.25,  # NutriNet-Sante cohort (Srour et al., BMJ 2019)
        "standard": 1.08,
        "mediterranean": 1.00,
        "dash": 1.00,
        "plant_based": 1.00,
    }
    sdoh_hr *= diet_hr_map.get(data.get("diet", "standard"), 1.00)

    # Exact Cox proportional hazards survival update: 1 - (1 - P_base)^HR_SDOH
    adjusted_prob = 1.0 - ((1.0 - base_prob) ** sdoh_hr)
    adjusted_prob = min(max(adjusted_prob, base_prob), 0.999)

    return base_prob * 100.0, adjusted_prob * 100.0


def verify_with_ml_model(data):
    """
    Runs the trained CDC NHANES XGBoost model on both the patient's biological baseline
    (ideal SDOH reference) and their true SDOH profile to independently verify the
    clinical formula outputs.
    """
    if not model:
        return None, None

    df_actual = map_form_to_features(data)

    df_baseline = df_actual.copy()
    df_baseline["IncomeBracket"] = 5.0
    df_baseline["HealthcareAccess"] = 0.0
    df_baseline["FoodInsecurity"] = 0.0
    df_baseline["DietQuality"] = 1.0
    df_baseline["Education"] = 5.0

    ml_base_risk = float(model.predict_proba(df_baseline)[0][1] * 100.0)
    ml_adjusted_risk = float(model.predict_proba(df_actual)[0][1] * 100.0)
    return ml_base_risk, ml_adjusted_risk


def generate_recommendations(risk):
    if risk < 5.0:
        return {
            "tier": "Low Risk",
            "action": "Maintain healthy lifestyle behaviors.",
            "clinical": "Statin therapy generally not indicated.",
        }
    elif 5.0 <= risk < 7.5:
        return {
            "tier": "Borderline Risk",
            "action": "Intensify lifestyle modifications.",
            "clinical": "Consider a CAC scan. Discuss moderate-intensity statins.",
        }
    elif 7.5 <= risk < 20.0:
        return {
            "tier": "Intermediate Risk",
            "action": "Immediate lifestyle modifications required.",
            "clinical": "Moderate to high-intensity statin therapy strongly recommended.",
        }
    else:
        return {
            "tier": "High Risk",
            "action": "Urgent risk factor modification.",
            "clinical": "High-intensity statin therapy indicated. Urgent cardiology referral.",
        }


def generate_ai_protocol(data, base_risk, adjusted_risk, ml_base_risk, ml_adjusted_risk):
    """Uses Gemini to verify clinical formula vs. NHANES ML model risks and generate a clinical summary."""
    if not gemini_client:
        return "AI Protocol unavailable. Please configure your GEMINI_API_KEY in the .env file."

    ml_context = (
        f"CDC NHANES XGBoost Verification -> Biological Baseline: {ml_base_risk}%, "
        f"SDOH-Adjusted: {ml_adjusted_risk}% (ML Equity Gap: +{round(ml_adjusted_risk - ml_base_risk, 1)}%)."
        if ml_base_risk is not None
        else "CDC NHANES XGBoost model file unavailable."
    )

    prompt = f"""
    You are an AI Clinical Navigator assessing a patient's cardiovascular risk.

    1. Official Medical Formula (ACC/AHA Pooled Cohort Equations + Cox SDOH Hazard Ratios):
       - Standard Biological Risk: {base_risk}%
       - Context-Adjusted Risk: {adjusted_risk}%
       - Formula Equity Gap: +{round(adjusted_risk - base_risk, 1)}%
    2. Empirical Machine Learning Verification ({ml_context})
    3. Patient Profile:
       - Age: {data.get('age')}, Gender: {data.get('gender')}, BP: {data.get('sysbp')} mmHg, Total Chol: {data.get('chol')} mg/dL, Diabetes: {data.get('diabetes')}, Smoker: {data.get('smoker')}, Conditions: {data.get('prev_diseases')}
       - SDOH Inputs: Income: {data.get('income')}, Diet: {data.get('diet')}, Healthcare Access: {data.get('healthcare')}, Food Access: {data.get('food_access')}.

    First, verify if the results make sense physically (risks must be between 0% and 100%, and Context-Adjusted Risk >= Standard Risk). If impossible, return "Error: Risk parameters out of bounds."

    Then, write a concise, professional clinical summary (3-4 sentences max):
    - State how the empirical CDC NHANES XGBoost model ({ml_base_risk}% -> {ml_adjusted_risk}%) compares with and corroborates the ACC/AHA clinical formula ({base_risk}% -> {adjusted_risk}%).
    - If the Adjusted Risk is HIGHER than the Standard Risk: Explain which specific structural SDOH barriers or clinical risk enhancers are driving the equity gap and suggest one concrete clinic-level intervention (e.g., social worker navigation, food pharmacy voucher, or patient assistance program for statins).
    - If the risks are the SAME (0% gap): Acknowledge that the patient's socioeconomic environment is protective and advise focusing directly on biological and lifestyle factors.

    Do not use bullet points or bold text. Keep the tone medical and empathetic.
    """
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )
        return response.text
    except Exception as e:
        error_msg = str(e).lower()
        if any(
            keyword in error_msg
            for keyword in ["429", "503", "demand", "quota", "exhausted"]
        ):
            return "The AI model is currently experiencing high demand. Please try generating the protocol again in a few moments."
        return f"Error generating protocol: {str(e)}"


@app.route("/")
def index():
    return render_template("index.html")


def validate_clinical_inputs(data):
    """
    Checks physiological inputs against hard boundaries to prevent
    impossible data from breaking the ML model or math functions.
    Returns a list of error strings if any are found.
    """
    errors = []

    try:
        age = float(data.get("age", 55))
        sysbp = float(data.get("sysbp", 120))
        chol = float(data.get("chol", 200))

        if age < 18 or age > 110:
            errors.append("Age must be between 18 and 110.")

        if sysbp < 70 or sysbp > 350:
            errors.append(
                f"Systolic BP of {sysbp} is outside valid physiological bounds (70-350 mmHg). Please verify."
            )

        if chol < 80 or chol > 600:
            errors.append(
                f"Total Cholesterol of {chol} is outside valid bounds (80-600 mg/dL). Please verify."
            )

    except (ValueError, TypeError):
        errors.append("Vitals must be valid numeric values.")

    return errors


@app.route("/calculate", methods=["POST"])
def calculate():
    data = request.json or {}

    # 1. Check physiological constraints
    validation_errors = validate_clinical_inputs(data)
    if validation_errors:
        error_message = " | ".join(validation_errors)
        return jsonify({"error": error_message}), 400

    # 2. Calculate risk percentages using official ACC/AHA & Cox medical formulas
    base_risk, adjusted_risk = calculate_clinical_formulas(data)

    # 3. Verify and compare with empirical CDC NHANES XGBoost model
    ml_base_risk, ml_adjusted_risk = verify_with_ml_model(data)

    base_r = round(base_risk, 1)
    adj_r = round(adjusted_risk, 1)
    gap_r = round(adjusted_risk - base_risk, 1)

    ml_base_r = round(ml_base_risk, 1) if ml_base_risk is not None else None
    ml_adj_r = round(ml_adjusted_risk, 1) if ml_adjusted_risk is not None else None
    ml_gap_r = (
        round(ml_adjusted_risk - ml_base_risk, 1)
        if (ml_base_risk is not None and ml_adjusted_risk is not None)
        else None
    )

    # 4. Generate AI clinical verification & summary via Gemini API
    ai_protocol = generate_ai_protocol(data, base_r, adj_r, ml_base_r, ml_adj_r)

    return jsonify(
        {
            "base_risk": base_r,
            "adjusted_risk": adj_r,
            "risk_gap": gap_r,
            "ml_base_risk": ml_base_r,
            "ml_adjusted_risk": ml_adj_r,
            "ml_risk_gap": ml_gap_r,
            "recommendations": generate_recommendations(adjusted_risk),
            "ai_protocol": ai_protocol,
        }
    )


@app.route("/chat", methods=["POST"])
def chat():
    if not gemini_client:
        return jsonify({"response": "Gemini API key is missing."}), 500

    data = request.json or {}
    user_message = data.get("message", "")
    system_context = (
        "You are an AI Clinical Navigator embedded in a Cardiovascular Risk Calculator "
        "that combines the ACC/AHA Pooled Cohort Equations with a CDC NHANES XGBoost verification model. "
        "Keep responses concise and medical."
    )
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash", contents=[system_context, user_message]
        )
        return jsonify({"response": response.text})
    except Exception as e:
        error_msg = str(e).lower()
        if any(
            keyword in error_msg
            for keyword in ["429", "503", "demand", "quota", "exhausted"]
        ):
            return jsonify(
                {
                    "response": "I am currently experiencing high demand. Please try asking again in a minute."
                }
            )
        return jsonify({"response": f"Error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
