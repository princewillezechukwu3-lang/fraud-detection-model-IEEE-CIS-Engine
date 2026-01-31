import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import joblib
from features import preprocess_transaction

app = Flask(__name__)
model =  joblib.load(r"C:\Users\ADMIN\ml projects\fraud detection project\models\fraud_model_v3.pkl")


FRAUD_THRESHOLD = 0.30
@app.route('/predict', methods=['POST'])
def predict():
    try:
        transaction = request.get_json()
        if not transaction:
            return jsonify({"error": "No data provided"}), 400
    
        raw_df = pd.DataFrame([transaction])
    
        preprocess_df = preprocess_transaction(raw_df)
        prob = float(model.predict(preprocess_df)[0])
        result = 'Fraud' if prob >= FRAUD_THRESHOLD else 'Not Fraud'

        if prob < 0.10:
            bucket = 'Low'
        elif prob < 0.50:
            bucket = 'Medium'
        else:
            bucket = 'High'

        return jsonify({
            'Result': result,
            'Probability': round(prob, 4),
            'Risk bucket': bucket
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        })


if __name__ == '__main__':
    app.run(debug=True, port=5000)