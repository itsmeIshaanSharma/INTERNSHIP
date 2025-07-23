from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)

# Global variables for model and scaler
model = None
scaler = None

def load_model():
    """Load the trained model and scaler"""
    global model, scaler
    try:
        # Try to load existing model
        if os.path.exists('salary_prediction_model.pkl'):
            with open('salary_prediction_model.pkl', 'rb') as file:
                model = pickle.load(file)
        
        if os.path.exists('scaler.pkl'):
            with open('scaler.pkl', 'rb') as file:
                scaler = pickle.load(file)
                
        print("Model and scaler loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.route('/')
def home():
    """Home page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Employee Salary Prediction</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .form-group { margin-bottom: 15px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input, select { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
            button { background-color: #007bff; color: white; padding: 12px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
            button:hover { background-color: #0056b3; }
            .result { margin-top: 20px; padding: 15px; border-radius: 5px; }
            .high-income { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .low-income { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Employee Salary Prediction</h1>
            <p>Enter the employee details below to predict if their annual income exceeds $50,000:</p>
            
            <form id="predictionForm">
                <div class="form-group">
                    <label for="age">Age:</label>
                    <input type="number" id="age" name="age" min="17" max="90" required>
                </div>
                
                <div class="form-group">
                    <label for="workclass">Work Class:</label>
                    <select id="workclass" name="workclass" required>
                        <option value="">Select Work Class</option>
                        <option value="0">Federal-gov</option>
                        <option value="1">Local-gov</option>
                        <option value="2">Never-worked</option>
                        <option value="3">Private</option>
                        <option value="4">Self-emp-inc</option>
                        <option value="5">Self-emp-not-inc</option>
                        <option value="6">State-gov</option>
                        <option value="7">Without-pay</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="educational_num">Years of Education:</label>
                    <input type="number" id="educational_num" name="educational_num" min="1" max="16" required>
                </div>
                
                <div class="form-group">
                    <label for="marital_status">Marital Status:</label>
                    <select id="marital_status" name="marital_status" required>
                        <option value="">Select Marital Status</option>
                        <option value="0">Divorced</option>
                        <option value="1">Married-AF-spouse</option>
                        <option value="2">Married-civ-spouse</option>
                        <option value="3">Married-spouse-absent</option>
                        <option value="4">Never-married</option>
                        <option value="5">Separated</option>
                        <option value="6">Widowed</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="hours_per_week">Hours per Week:</label>
                    <input type="number" id="hours_per_week" name="hours_per_week" min="1" max="99" required>
                </div>
                
                <div class="form-group">
                    <label for="gender">Gender:</label>
                    <select id="gender" name="gender" required>
                        <option value="">Select Gender</option>
                        <option value="0">Female</option>
                        <option value="1">Male</option>
                    </select>
                </div>
                
                <button type="submit">Predict Salary</button>
            </form>
            
            <div id="result"></div>
        </div>
        
        <script>
            document.getElementById('predictionForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData(this);
                const data = {
                    age: parseInt(formData.get('age')),
                    workclass: parseInt(formData.get('workclass')),
                    fnlwgt: 200000, // Default value
                    educational_num: parseInt(formData.get('educational_num')),
                    marital_status: parseInt(formData.get('marital_status')),
                    occupation: 1, // Default value
                    relationship: 1, // Default value
                    race: 4, // Default value
                    gender: parseInt(formData.get('gender')),
                    capital_gain: 0, // Default value
                    capital_loss: 0, // Default value
                    hours_per_week: parseInt(formData.get('hours_per_week')),
                    native_country: 39 // Default value (United States)
                };
                
                try {
                    const response = await fetch('/api/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    
                    if (result.error) {
                        document.getElementById('result').innerHTML = 
                            '<div class="result" style="background-color: #f8d7da; color: #721c24;">Error: ' + result.error + '</div>';
                    } else {
                        const isHighIncome = result.prediction === '>50K';
                        const confidence = (result.confidence * 100).toFixed(2);
                        
                        document.getElementById('result').innerHTML = 
                            '<div class="result ' + (isHighIncome ? 'high-income' : 'low-income') + '">' +
                            '<h3>Prediction Result:</h3>' +
                            '<p><strong>Predicted Income:</strong> ' + result.prediction + '</p>' +
                            '<p><strong>Confidence:</strong> ' + confidence + '%</p>' +
                            '</div>';
                    }
                } catch (error) {
                    document.getElementById('result').innerHTML = 
                        '<div class="result" style="background-color: #f8d7da; color: #721c24;">Error: ' + error.message + '</div>';
                }
            });
        </script>
    </body>
    </html>
    """

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        if model is None or scaler is None:
            return jsonify({'error': 'Model not loaded. Please ensure model files exist.'}), 500
            
        data = request.get_json()
        
        # Create feature array in the correct order
        features = np.array([[
            data['age'],
            data['workclass'],
            data.get('fnlwgt', 200000),
            data['educational_num'],
            data['marital_status'],
            data.get('occupation', 1),
            data.get('relationship', 1),
            data.get('race', 4),
            data['gender'],
            data.get('capital_gain', 0),
            data.get('capital_loss', 0),
            data['hours_per_week'],
            data.get('native_country', 39)
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        return jsonify({
            'prediction': ">50K" if prediction == 1 else "<=50K",
            'confidence': float(max(probability))
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })

if __name__ == '__main__':
    # Load model on startup
    if not load_model():
        print("Warning: Could not load model files. Please ensure 'salary_prediction_model.pkl' and 'scaler.pkl' exist.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
