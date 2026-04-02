"""
FastAPI Application for Telco Customer Churn Prediction
Provides REST API endpoints for making predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pickle
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using machine learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL = None
FEATURE_COLUMNS = None


# Pydantic models for request/response
class CustomerInput(BaseModel):
    """Single customer input for prediction"""
    gender: int
    SeniorCitizen: float
    Partner: int
    Dependents: int
    tenure: float
    PhoneService: int
    MultipleLines: int
    PaperlessBilling: int
    MonthlyCharges: float
    TotalCharges: float
    InternetService_Fiber_optic: bool
    InternetService_No: bool
    OnlineSecurity_No_internet_service: bool
    OnlineSecurity_Yes: bool
    OnlineBackup_No_internet_service: bool
    OnlineBackup_Yes: bool
    DeviceProtection_No_internet_service: bool
    DeviceProtection_Yes: bool
    TechSupport_No_internet_service: bool
    TechSupport_Yes: bool
    StreamingTV_No_internet_service: bool
    StreamingTV_Yes: bool
    StreamingMovies_No_internet_service: bool
    StreamingMovies_Yes: bool
    Contract_One_year: bool
    Contract_Two_year: bool
    PaymentMethod_Credit_card_automatic: bool
    PaymentMethod_Electronic_check: bool
    PaymentMethod_Mailed_check: bool


class PredictionResponse(BaseModel):
    """Prediction response"""
    prediction: str
    churn_probability: float
    no_churn_probability: float
    risk_level: str
    confidence: float


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    total_predictions: int
    churn_count: int
    no_churn_count: int
    churn_percentage: float
    predictions: List[dict]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    version: str


class FeatureImportanceResponse(BaseModel):
    """Feature importance response"""
    top_features: List[dict]
    total_features: int


def load_model():
    """Load the trained model"""
    global MODEL, FEATURE_COLUMNS
    
    # Try multiple possible paths
    possible_paths = [
        'models/best_model.pkl',
        './models/best_model.pkl',
        '/Users/jinka.sai21/Documents/Python/Customer-Churn-Prediction/models/best_model.pkl'
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print(f"✗ Model file not found in any of these locations: {possible_paths}")
        return False
    
    try:
        with open(model_path, 'rb') as f:
            MODEL = pickle.load(f)
        
        # Try multiple paths for processed data
        data_paths = [
            'data/X_processed.csv',
            './data/X_processed.csv',
            '/Users/jinka.sai21/Documents/Python/Customer-Churn-Prediction/data/X_processed.csv'
        ]
        
        X = None
        for path in data_paths:
            if os.path.exists(path):
                X = pd.read_csv(path)
                break
        
        if X is not None:
            FEATURE_COLUMNS = X.columns.tolist()
        
        print(f"✓ Model loaded successfully from {model_path}")
        print(f"✓ Features: {len(FEATURE_COLUMNS)} columns")
        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False


def map_feature_names(data_dict):
    """Map feature names from underscores to match model training"""
    mapping = {
        'InternetService_Fiber_optic': 'InternetService_Fiber optic',
        'OnlineSecurity_No_internet_service': 'OnlineSecurity_No internet service',
        'OnlineBackup_No_internet_service': 'OnlineBackup_No internet service',
        'DeviceProtection_No_internet_service': 'DeviceProtection_No internet service',
        'TechSupport_No_internet_service': 'TechSupport_No internet service',
        'StreamingTV_No_internet_service': 'StreamingTV_No internet service',
        'StreamingMovies_No_internet_service': 'StreamingMovies_No internet service',
        'Contract_One_year': 'Contract_One year',
        'Contract_Two_year': 'Contract_Two year',
        'PaymentMethod_Credit_card_automatic': 'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic_check': 'PaymentMethod_Electronic check',
        'PaymentMethod_Mailed_check': 'PaymentMethod_Mailed check',
    }
    
    mapped_dict = {}
    for key, value in data_dict.items():
        new_key = mapping.get(key, key)
        mapped_dict[new_key] = value
    
    return mapped_dict


def get_risk_level(churn_probability):
    """Determine risk level based on churn probability"""
    if churn_probability > 0.6:
        return "High"
    elif churn_probability > 0.3:
        return "Medium"
    else:
        return "Low"


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    print("Starting up...")
    load_model()


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "openapi": "/openapi.json"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if MODEL is not None else "unhealthy",
        "model_loaded": MODEL is not None,
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(customer: CustomerInput):
    """
    Make a prediction for a single customer
    
    Returns:
        - prediction: "Churn" or "No Churn"
        - churn_probability: Probability of churn (0-1)
        - no_churn_probability: Probability of no churn (0-1)
        - risk_level: "Low", "Medium", or "High"
        - confidence: Confidence level of prediction (0-1)
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to dict
        data_dict = customer.dict()
        
        # Map feature names
        data_dict = map_feature_names(data_dict)
        
        # Create DataFrame
        df = pd.DataFrame([data_dict])
        
        # Ensure all required columns are present in correct order
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0
        
        df = df[FEATURE_COLUMNS]
        
        # Make prediction
        prediction = MODEL.predict(df)[0]
        probabilities = MODEL.predict_proba(df)[0]
        
        churn_prob = float(probabilities[1])
        no_churn_prob = float(probabilities[0])
        risk_level = get_risk_level(churn_prob)
        
        return {
            "prediction": "Churn" if prediction == 1 else "No Churn",
            "churn_probability": churn_prob,
            "no_churn_probability": no_churn_prob,
            "risk_level": risk_level,
            "confidence": max(churn_prob, no_churn_prob)
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(file_path: Optional[str] = None):
    """
    Make predictions for a batch of customers
    
    Args:
        file_path: Path to CSV file with customer data (optional)
                   If not provided, uses data/X_processed.csv
    
    Returns:
        - total_predictions: Total number of predictions made
        - churn_count: Number of customers predicted to churn
        - no_churn_count: Number of customers predicted not to churn
        - churn_percentage: Percentage of customers predicted to churn
        - predictions: List of predictions with probabilities
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Use provided file or default
        input_file = file_path if file_path else 'data/X_processed.csv'
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"File not found: {input_file}")
        
        # Load data
        X = pd.read_csv(input_file)
        
        # Make predictions
        predictions = MODEL.predict(X)
        probabilities = MODEL.predict_proba(X)
        
        # Prepare results
        churn_count = int((predictions == 1).sum())
        no_churn_count = int((predictions == 0).sum())
        total = len(predictions)
        churn_percentage = (churn_count / total * 100) if total > 0 else 0
        
        # Create prediction list (limit to first 100 for API response)
        prediction_list = []
        for i in range(min(100, len(predictions))):
            churn_prob = float(probabilities[i][1])
            prediction_list.append({
                "customer_id": i,
                "prediction": "Churn" if predictions[i] == 1 else "No Churn",
                "churn_probability": churn_prob,
                "no_churn_probability": float(probabilities[i][0]),
                "risk_level": get_risk_level(churn_prob)
            })
        
        return {
            "total_predictions": total,
            "churn_count": churn_count,
            "no_churn_count": no_churn_count,
            "churn_percentage": churn_percentage,
            "predictions": prediction_list
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")


@app.get("/features", tags=["Model Info"])
async def get_features():
    """Get list of features used by the model"""
    if FEATURE_COLUMNS is None:
        raise HTTPException(status_code=503, detail="Features not loaded")
    
    return {
        "total_features": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS
    }


@app.get("/feature-importance", response_model=FeatureImportanceResponse, tags=["Model Info"])
async def get_feature_importance():
    """Get feature importance from the model"""
    if MODEL is None or not hasattr(MODEL, 'coef_'):
        raise HTTPException(status_code=400, detail="Feature importance not available")
    
    try:
        # Get coefficients
        coefficients = MODEL.coef_[0]
        
        # Create importance list
        importance_list = []
        for feat, coef in zip(FEATURE_COLUMNS, coefficients):
            importance_list.append({
                "feature": feat,
                "coefficient": float(coef),
                "abs_coefficient": float(abs(coef))
            })
        
        # Sort by absolute coefficient
        importance_list.sort(key=lambda x: x['abs_coefficient'], reverse=True)
        
        return {
            "top_features": importance_list[:10],  # Top 10 features
            "total_features": len(importance_list)
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")


@app.get("/model-info", tags=["Model Info"])
async def get_model_info():
    """Get information about the trained model"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(MODEL).__name__,
        "total_features": len(FEATURE_COLUMNS),
        "model_path": "models/best_model.pkl",
        "dataset_size": "7043 samples",
        "test_accuracy": "80.34%",
        "roc_auc_score": "0.8427",
        "status": "production"
    }


@app.get("/example-input", tags=["Examples"])
async def get_example_input():
    """Get example input for prediction"""
    return {
        "description": "Example customer input for churn prediction",
        "example": {
            "gender": 1,
            "SeniorCitizen": -0.44,
            "Partner": 1,
            "Dependents": 0,
            "tenure": 0.93,
            "PhoneService": 1,
            "MultipleLines": 1,
            "PaperlessBilling": 1,
            "MonthlyCharges": 0.0,
            "TotalCharges": -1.0,
            "InternetService_Fiber_optic": False,
            "InternetService_No": True,
            "OnlineSecurity_No_internet_service": True,
            "OnlineSecurity_Yes": False,
            "OnlineBackup_No_internet_service": True,
            "OnlineBackup_Yes": False,
            "DeviceProtection_No_internet_service": True,
            "DeviceProtection_Yes": False,
            "TechSupport_No_internet_service": True,
            "TechSupport_Yes": False,
            "StreamingTV_No_internet_service": True,
            "StreamingTV_Yes": False,
            "StreamingMovies_No_internet_service": True,
            "StreamingMovies_Yes": False,
            "Contract_One_year": False,
            "Contract_Two_year": False,
            "PaymentMethod_Credit_card_automatic": False,
            "PaymentMethod_Electronic_check": False,
            "PaymentMethod_Mailed_check": False
        }
    }


if __name__ == '__main__':
    import uvicorn
    
    print("="*60)
    print("Starting Customer Churn Prediction API")
    print("="*60)
    print("\nAPI Documentation: http://localhost:8000/docs")
    print("Alternative Docs: http://localhost:8000/redoc")
    print("\n" + "="*60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
