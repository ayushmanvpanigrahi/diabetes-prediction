from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('random_forest_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        features = np.array([[
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['bloodpressure']),
            float(request.form['skinthickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['diabetespedigree']),
            float(request.form['age'])
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        # Create response
        result = {
            'prediction': int(prediction),
            'probability': round(float(probability) * 100, 2),
            'message': "High Risk of Diabetes" if prediction == 1 else "Low Risk of Diabetes"
        }
        
        return render_template('result.html', result=result)
        
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
