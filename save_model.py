"""
Script to train and save the salary prediction model
Run this script to generate the required .pkl files for the Flask app
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle

def train_and_save_model():
    """Train the model and save it along with the scaler"""
    
    print("Loading and preprocessing data...")
    
    # Note: You'll need to replace this with your actual data loading
    # This is a placeholder - use your existing notebook's data preprocessing
    
    # Load your data here
    # data = pd.read_csv('adult.csv')  # Replace with your actual data file
    
    # For demonstration, creating sample data structure
    # In your case, use the preprocessed data from your notebook
    
    print("Creating sample data for demonstration...")
    # This is just for demonstration - replace with your actual data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = {
        'age': np.random.randint(17, 91, n_samples),
        'workclass': np.random.randint(0, 8, n_samples),
        'fnlwgt': np.random.randint(50000, 500000, n_samples),
        'educational_num': np.random.randint(1, 17, n_samples),
        'marital_status': np.random.randint(0, 7, n_samples),
        'occupation': np.random.randint(0, 15, n_samples),
        'relationship': np.random.randint(0, 6, n_samples),
        'race': np.random.randint(0, 5, n_samples),
        'gender': np.random.randint(0, 2, n_samples),
        'capital_gain': np.random.randint(0, 10000, n_samples),
        'capital_loss': np.random.randint(0, 5000, n_samples),
        'hours_per_week': np.random.randint(1, 100, n_samples),
        'native_country': np.random.randint(0, 40, n_samples),
        'income': np.random.choice(['<=50K', '>50K'], n_samples)
    }
    
    data = pd.DataFrame(sample_data)
    
    # Prepare features and target
    X = data.drop('income', axis=1)
    y = data['income']
    
    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print("Training the model...")
    
    # Initialize and fit the scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the Gradient Boosting model (best performer from your analysis)
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))
    
    # Save the model and scaler
    print("Saving model and scaler...")
    
    with open('salary_prediction_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Model and scaler saved successfully!")
    print("Files created:")
    print("- salary_prediction_model.pkl")
    print("- scaler.pkl")
    
    return model, scaler

if __name__ == "__main__":
    train_and_save_model()
