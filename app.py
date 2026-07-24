from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


def calculate_mock_ascvd(data):
    """
    A simplified baseline calculation simulating standard ASCVD equations.
    In reality, you would plug in the Pooled Cohort Equations or your ML model here.
    """
    age = float(data.get('age', 50))
    sys_bp = float(data.get('sysbp', 120))
    chol = float(data.get('chol', 200))
    smoker = data.get('smoker', 'no') == 'yes'

    # Baseline rough calculation
    risk = (age / 65.0) * (sys_bp / 120.0) * (chol / 180.0) * 4.5
    if smoker:
        risk *= 1.5

    return min(risk, 100.0)


def apply_sdoh_shift(base_risk, data):
    """
    Demonstrates the 'invisible risk' by adjusting the score based on
    Social Determinants of Health (SDOH).
    """
    income = data.get('income', 'high')
    food_access = data.get('food_access', 'high')
    healthcare_access = data.get('healthcare_access', 'good')

    multiplier = 1.0

    # These represent the unseen structural risks standard models miss
    if income == 'low':
        multiplier += 0.25  # Lower income correlates with higher allostatic load
    if food_access == 'low':
        multiplier += 0.15  # Food deserts impact diet quality and BP
    if healthcare_access == 'poor':
        multiplier += 0.30  # Poor access means delayed interventions

    return min(base_risk * multiplier, 100.0)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.json

    base_risk = calculate_mock_ascvd(data)
    adjusted_risk = apply_sdoh_shift(base_risk, data)

    return jsonify({
        'base_risk': round(base_risk, 1),
        'adjusted_risk': round(adjusted_risk, 1),
        'risk_gap': round(adjusted_risk - base_risk, 1)
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)