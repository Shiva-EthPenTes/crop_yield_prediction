import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Synthetic data generation for demonstration
def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'temperature': np.random.uniform(15, 35, n_samples),  # Celsius
        'rainfall': np.random.uniform(50, 200, n_samples),   # mm/month
        'soil_ph': np.random.uniform(5.5, 7.5, n_samples),
        'soil_moisture': np.random.uniform(20, 80, n_samples),  # %
        'nitrogen': np.random.uniform(10, 50, n_samples),      # kg/ha
        'irrigation_frequency': np.random.randint(1, 5, n_samples),  # times/week
        'yield': np.zeros(n_samples)  # Target variable (tons/ha)
    }
    df = pd.DataFrame(data)
    # Simple synthetic yield calculation (replace with real data)
    df['yield'] = (
        0.1 * df['temperature'] +
        0.05 * df['rainfall'] +
        0.2 * df['soil_ph'] +
        0.3 * df['soil_moisture'] +
        0.15 * df['nitrogen'] +
        0.5 * df['irrigation_frequency'] +
        np.random.normal(0, 0.5, n_samples)
    )
    return df

# Data preprocessing
def preprocess_data(df):
    try:
        X = df[['temperature', 'rainfall', 'soil_ph', 'soil_moisture', 'nitrogen', 'irrigation_frequency']]
        y = df['yield']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X, y, scaler
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise

# Train model
def train_model(X, y):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logger.info(f"Model Performance - MAE: {mae:.2f}, R2: {r2:.2f}")
        
        return model
    except Exception as e:
        logger.error(f"Error in training: {e}")
        raise

# Save model and scaler
def save_artifacts(model, scaler, model_path="model.pkl", scaler_path="scaler.pkl"):
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info("Model and scaler saved successfully")
    except Exception as e:
        logger.error(f"Error saving artifacts: {e}")
        raise

# Optimization function (simple heuristic-based recommendation)
def optimize_practices(input_data):
    recommendations = {}
    if input_data['soil_moisture'] < 30:
        recommendations['irrigation'] = "Increase irrigation to 3-4 times/week"
    if input_data['soil_ph'] < 6.0:
        recommendations['soil'] = "Apply lime to increase soil pH"
    if input_data['nitrogen'] < 20:
        recommendations['fertilizer'] = "Apply 30-40 kg/ha nitrogen fertilizer"
    return recommendations

# FastAPI application
app = FastAPI(title="Crop Yield Prediction API")

# Input schema
class FarmInput(BaseModel):
    temperature: float
    rainfall: float
    soil_ph: float
    soil_moisture: float
    nitrogen: float
    irrigation_frequency: int

# Load model and scaler
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    # Train model if artifacts don't exist
    df = generate_synthetic_data()
    X, y, scaler = preprocess_data(df)
    model = train_model(X, y)
    save_artifacts(model, scaler)

# API endpoint for prediction and optimization
@app.post("/predict")
async def predict_yield(input: FarmInput):
    try:
        # Prepare input data
        input_data = pd.DataFrame([input.dict()])
        X_scaled = scaler.transform(input_data)
        
        # Predict yield
        prediction = model.predict(X_scaled)[0]
        
        # Generate recommendations
        recommendations = optimize_practices(input.dict())
        
        return {
            "yield_prediction": round(float(prediction), 2),
            "recommendations": recommendations
        }
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return {"error": str(e)}

# Main execution for training (run once to generate artifacts)
if __name__ == "__main__":
    import uvicorn
    # Generate and preprocess data
    df = generate_synthetic_data()
    X, y, scaler = preprocess_data(df)
    
    # Train and save model
    model = train_model(X, y)
    save_artifacts(model, scaler)
    
    # Run FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)