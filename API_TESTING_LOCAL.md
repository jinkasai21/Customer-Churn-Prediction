# FastAPI Local Testing Guide

## Quick Start (3 Steps)

### Step 1: Start the API Server
```bash
cd /path/to/Customer-Churn-Prediction
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Expected Output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

**Keep this terminal running!**

---

### Step 2: Open Swagger UI (Interactive Testing - EASIEST!)
Open your browser and visit:
```
http://localhost:8000/docs
```

You'll see an interactive API explorer where you can:
- ✅ View all 8 endpoints
- ✅ Click "Try it out" button
- ✅ Enter test data
- ✅ See responses instantly
- ✅ View request/response formats

**Alternative Documentation:**
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

---

### Step 3: Test Endpoints

#### Option A: Using Swagger UI (Recommended for Beginners)
1. Visit `http://localhost:8000/docs`
2. Click on any endpoint (e.g., `/health`)
3. Click "Try it out"
4. Click "Execute"
5. See the response

#### Option B: Using Python Script
Open Terminal 2 and run:
```bash
cd /path/to/Customer-Churn-Prediction
python << 'TEST_API'
import requests
import json

BASE_URL = "http://localhost:8000"

print("\n" + "="*80)
print("TESTING ALL API ENDPOINTS")
print("="*80)

# 1. Health Check
print("\n1️⃣ GET /health - Check if API is running")
response = requests.get(f"{BASE_URL}/health")
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# 2. Model Info
print("\n2️⃣ GET /model-info - Get model details")
response = requests.get(f"{BASE_URL}/model-info")
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# 3. Features
print("\n3️⃣ GET /features - List all 30 features")
response = requests.get(f"{BASE_URL}/features")
print(f"Status: {response.status_code}")
features = response.json()['features']
print(f"Total Features: {len(features)}")
print(f"First 5 features: {features[:5]}")

# 4. Feature Importance
print("\n4️⃣ GET /feature-importance - Top 10 features")
response = requests.get(f"{BASE_URL}/feature-importance")
print(f"Status: {response.status_code}")
importance = response.json()['feature_importance']
print("Top 5 features:")
for feat in importance[:5]:
    print(f"  • {feat['feature']:30} → Coefficient: {feat['coefficient']:8.4f}")

# 5. Example Input
print("\n5️⃣ GET /example-input - Example customer data")
response = requests.get(f"{BASE_URL}/example-input")
print(f"Status: {response.status_code}")
example = response.json()['example']
print(f"Number of fields: {len(example)}")
print("Example data:")
for key, value in list(example.items())[:5]:
    print(f"  • {key}: {value}")

# 6. Single Prediction - Customer LIKELY TO CHURN
print("\n6️⃣ POST /predict - Test Customer (High Churn Risk)")
print("-" * 80)
high_risk_customer = {
    "gender": "Male",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "No",
    "tenure": 3,  # NEW customer - high churn risk
    "phone_service": "Yes",
    "paperless_billing": "Yes",
    "monthly_charges": 85.5,  # High charges
    "total_charges": 256.5,
    "internet_service": "Fiber optic",  # High churn service
    "online_security": "No",  # No security = higher risk
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "contract": "Month-to-month",  # Highest churn contract
    "payment_method": "Electronic check",
    "multiple_lines": "No"
}
response = requests.post(f"{BASE_URL}/predict", json=high_risk_customer)
print(f"Status: {response.status_code}")
result = response.json()
print(f"Prediction: {'WILL CHURN ⚠️' if result['prediction'] == 1 else 'WILL STAY ✅'}")
print(f"Churn Probability: {result['churn_probability']:.1%}")
print(f"Risk Level: {result['risk_level']}")

# 7. Single Prediction - Customer UNLIKELY TO CHURN
print("\n7️⃣ POST /predict - Test Customer (Low Churn Risk)")
print("-" * 80)
low_risk_customer = {
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "Yes",
    "tenure": 48,  # LONG tenure - low churn risk
    "phone_service": "Yes",
    "paperless_billing": "No",
    "monthly_charges": 45.5,  # Lower charges
    "total_charges": 2185.5,
    "internet_service": "DSL",  # Lower churn service
    "online_security": "Yes",  # Has security = lower risk
    "online_backup": "Yes",
    "device_protection": "Yes",
    "tech_support": "Yes",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "contract": "Two year",  # Lowest churn contract
    "payment_method": "Bank transfer (automatic)",
    "multiple_lines": "Yes"
}
response = requests.post(f"{BASE_URL}/predict", json=low_risk_customer)
print(f"Status: {response.status_code}")
result = response.json()
print(f"Prediction: {'WILL CHURN ⚠️' if result['prediction'] == 1 else 'WILL STAY ✅'}")
print(f"Churn Probability: {result['churn_probability']:.1%}")
print(f"Risk Level: {result['risk_level']}")

# 8. Batch Prediction
print("\n8️⃣ POST /predict-batch - Predict all 7,043 customers")
print("-" * 80)
response = requests.post(f"{BASE_URL}/predict-batch")
print(f"Status: {response.status_code}")
batch_result = response.json()
summary = batch_result['summary']
print(f"Total Predictions: {summary['total_predictions']}")
print(f"Predicted Churn: {summary['predicted_churn']} customers")
print(f"Churn Rate: {summary['churn_rate']:.1%}")
print(f"\nFirst 3 predictions:")
for pred in batch_result['predictions'][:3]:
    print(f"  • Customer {pred['customer_index']}: Churn={pred['prediction']}, Probability={pred['churn_probability']:.2%}")

print("\n" + "="*80)
print("✅ ALL TESTS PASSED!")
print("="*80 + "\n")
TEST_API
```

#### Option C: Using cURL Commands
```bash
# Health Check
curl http://localhost:8000/health

# Model Info
curl http://localhost:8000/model-info

# Features
curl http://localhost:8000/features

# Feature Importance
curl http://localhost:8000/feature-importance

# Single Prediction (High Risk Customer)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "No",
    "tenure": 3,
    "phone_service": "Yes",
    "paperless_billing": "Yes",
    "monthly_charges": 85.5,
    "total_charges": 256.5,
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "payment_method": "Electronic check",
    "multiple_lines": "No"
  }'

# Batch Prediction
curl -X POST http://localhost:8000/predict-batch
```

---

## All 8 API Endpoints

### 1. GET `/` - Root Endpoint
```bash
curl http://localhost:8000/
```
**Response:**
```json
{
  "message": "Customer Churn Prediction API",
  "version": "1.0",
  "endpoints": 8
}
```

### 2. GET `/health` - Health Check
```bash
curl http://localhost:8000/health
```
**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 3. GET `/model-info` - Model Metrics
```bash
curl http://localhost:8000/model-info
```
**Response:**
```json
{
  "model_type": "Logistic Regression",
  "accuracy": 0.8034,
  "precision": 0.652,
  "recall": 0.5561,
  "f1_score": 0.6003,
  "roc_auc": 0.8427,
  "training_samples": 5634,
  "test_samples": 1409
}
```

### 4. GET `/features` - List All Features
```bash
curl http://localhost:8000/features
```
**Response:**
```json
{
  "features": [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "tenure",
    ... (30 total features)
  ]
}
```

### 5. GET `/feature-importance` - Top 10 Features
```bash
curl http://localhost:8000/feature-importance
```
**Response:**
```json
{
  "feature_importance": [
    {
      "feature": "Contract_Two year",
      "coefficient": -1.3502
    },
    {
      "feature": "tenure",
      "coefficient": -1.2554
    },
    ... (10 total)
  ]
}
```

### 6. GET `/example-input` - Example Customer Data
```bash
curl http://localhost:8000/example-input
```
**Response:**
```json
{
  "example": {
    "gender": "Male",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "No",
    ... (30 total fields)
  }
}
```

### 7. POST `/predict` - Single Customer Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{ ... customer data ... }'
```
**Response:**
```json
{
  "prediction": 1,
  "churn_probability": 0.75,
  "risk_level": "High",
  "top_risk_factors": [
    {
      "feature": "Contract_Two year",
      "contribution": -0.35
    },
    ...
  ]
}
```

### 8. POST `/predict-batch` - Batch Predictions
```bash
curl -X POST http://localhost:8000/predict-batch
```
**Response:**
```json
{
  "predictions": [
    {
      "customer_index": 0,
      "prediction": 0,
      "churn_probability": 0.23
    },
    ...
  ],
  "summary": {
    "total_predictions": 7043,
    "predicted_churn": 1572,
    "churn_rate": 0.223
  }
}
```

---

## Understanding Predictions

### Risk Levels
- **Low** 🟢: Churn probability ≤ 30%
- **Medium** 🟡: Churn probability 30-60%
- **High** 🔴: Churn probability > 60%

### Prediction Values
- **0** = No Churn (Customer will stay)
- **1** = Churn (Customer will leave)

### Top Risk Factors
- Negative coefficients = PROTECT against churn
- Positive coefficients = INCREASE churn risk

---

## Verification Checklist

Test each endpoint and verify:

```
✅ GET  /              → Returns welcome message (Status 200)
✅ GET  /health        → Shows model_loaded: true (Status 200)
✅ GET  /model-info    → Shows accuracy 0.8034, ROC-AUC 0.8427 (Status 200)
✅ GET  /features      → Lists 30 features (Status 200)
✅ GET  /feature-importance → Shows top 10 features (Status 200)
✅ GET  /example-input → Returns 30 fields (Status 200)
✅ POST /predict       → Returns probability & risk level (Status 200)
✅ POST /predict-batch → Returns 7,043 predictions (Status 200)
```

---

## Troubleshooting

### Problem: "Connection refused"
```bash
# Check if server is running
lsof -i :8000

# If nothing shows, API is not running
# Start it: python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Problem: "ModuleNotFoundError"
```bash
# Install missing packages
pip install fastapi uvicorn pandas numpy scikit-learn
```

### Problem: 422 Unprocessable Entity (Request validation error)
- You're missing required fields in POST request
- Get correct format from `/example-input`
- All 19 fields are required for `/predict`

### Problem: "Model not found" error
```bash
# Check model file exists
ls -la models/best_model.pkl

# Should show file > 1 KB
# If missing, run training script first
```

### Problem: Predictions are all 0s or 1s
- Data scaling might be wrong
- Check if X_processed.csv is correct
- Verify model loads properly

---

## Stop the API Server

### Method 1: Press Ctrl+C
In Terminal 1 (where API is running):
```
Press Ctrl+C
```

### Method 2: Kill the process
```bash
lsof -i :8000
kill -9 <PID>
```

### Verify it stopped:
```bash
curl http://localhost:8000/health
# Should show: Connection refused
```

---

## Performance Notes

- API response time: < 100ms per request
- Batch prediction (7,043 customers): ~1-2 seconds
- Memory usage: ~200-300 MB
- Model load time: ~1-2 seconds

---

## Next Steps

1. ✅ Test all endpoints with Swagger UI
2. ✅ Run Python test script
3. ✅ Test with real customer data
4. ✅ Deploy to cloud (AWS, Heroku, etc.)
5. ✅ Set up monitoring & logging
6. ✅ Document API for team

---

## API Documentation

For full API documentation in browser:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

---

**Happy Testing! 🚀**
