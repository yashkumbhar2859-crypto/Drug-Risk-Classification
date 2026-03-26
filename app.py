from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model (pipeline)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    try:
        # 1️⃣ Collect numeric values
        numeric_values = []
        for i in range(1, 24):
            numeric_values.append(float(request.form[f'n{i}']))

        # 2️⃣ Collect categorical values
        drug_form = request.form['Drug_Form']
        therapeutic_class = request.form['Therapeutic_Class']
        manufacturing_region = request.form['Manufacturing_Region']
        requires_cold_storage = request.form['Requires_Cold_Storage']
        otc_flag = request.form['OTC_Flag']
        high_risk_substance = request.form['High_Risk_Substance']

        # 3️⃣ Combine everything in correct order
        input_data = numeric_values + [
            drug_form,
            therapeutic_class,
            manufacturing_region,
            requires_cold_storage,
            otc_flag,
            high_risk_substance
        ]

        # 4️⃣ Create DataFrame with correct column names
        columns = [
            # Replace these with your actual numeric column names in correct order
            # Example:
            # 'Dosage_mg', 'Shelf_Life_Months', ...
        ]

        # IMPORTANT: Instead of manual typing, we will auto-load column names
        columns = model.named_steps['preprocess'].feature_names_in_

        input_df = pd.DataFrame([input_data], columns=columns)

        # 5️⃣ Make prediction
        prediction = model.predict(input_df)

        return render_template('index.html',
                               prediction_text=f'Prediction: {prediction[0]}')

    except Exception as e:
        return render_template('index.html',
                               prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)