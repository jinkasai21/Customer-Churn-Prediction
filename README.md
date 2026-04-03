# Customer Churn Prediction - ML Pipeline & REST API

**Built an end-to-end machine learning pipeline to predict customer churn in telecommunications with real-time inference via FastAPI.**

### Key Highlights
- ✅ **Data Pipeline**: Implemented robust preprocessing with feature engineering (30 features from 21 raw columns)
- ✅ **Model Training**: Trained & compared multiple models (Logistic Regression, Random Forest); achieved **80.34% accuracy** with **0.8427 ROC-AUC**
- ✅ **Production API**: Exposed model via **FastAPI** with 8 endpoints for real-time predictions on single customers and batch processing
- ✅ **Modular Architecture**: Production-ready code with clear separation of concerns (preprocessing, training, inference, API)
- ✅ **Comprehensive Testing**: 6 testing methods + detailed guides for validation and deployment
- ✅ **Dataset**: 7,043 customer records with 26.5% churn rate

### Project Status
- 🚀 **Production Ready**: Fully functional with comprehensive documentation
- 📊 **Model Performance**: 80.34% accuracy, 85.27% ROC-AUC on test set
- 🌐 **API Deployed**: 8 REST endpoints with interactive Swagger UI
- 📝 **Well Documented**: TESTING_GUIDE.txt + API_TESTING_LOCAL.md for easy deployment

## Architecture Overview

```
Raw Data (7,043 customers, 21 features)
    ↓
Data Preprocessing (handle missing values, encode categorical, scale numerical)
    ↓
Feature Engineering (30 processed features)
    ↓
Model Training (Logistic Regression, Random Forest)
    ↓
Model Selection (Best: Logistic Regression - 80.34% accuracy)
    ↓
REST API (FastAPI with 8 endpoints)
    ↓
Real-time Predictions (single customer + batch processing)
```

## Project Structure

```
Customer-Churn-Prediction/
│
├── 📁 data/                              # Datasets
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv    (Raw: 7,043 × 21)
│   ├── X_processed.csv                          (Processed: 7,043 × 30)
│   ├── y_processed.csv                          (Target: 7,043 × 1)
│   └── predictions.csv                          (Model predictions)
│
├── 📁 models/                            # Trained models & artifacts
│   ├── best_model.pkl                    (Logistic Regression - 80.34% accuracy)
│   ├── model_metadata.txt                (Performance metrics)
│   ├── confusion_matrices.png            (Visualization)
│   └── model_comparison.png              (3-model comparison)
│
├── 📁 src/                               # ML pipeline modules
│   ├── data_preprocessing.py             (Load, clean, encode, scale data)
│   ├── train_model.py                    (Train 3 models, evaluate, compare)
│   ├── predict.py                        (Single & batch predictions)
│   └── utils/__init__.py
│
├── 📁 app/                               # FastAPI application
│   └── main.py                           (8 REST endpoints)
│
├── 📁 notebooks/                         # Jupyter notebooks (optional)
│
├── 📄 requirements.txt                   (Python dependencies)
├── 📄 README.md                          (This file)
├── 📄 TESTING_GUIDE.txt                  (6 testing methods)
└── 📄 API_TESTING_LOCAL.md               (API testing guide)
```

## Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/jinkasai21/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import pandas, numpy, sklearn, fastapi; print('✅ All packages installed!')"
```

## Dataset

**Telco Customer Churn** - Real-world telecommunications customer dataset

| Metric | Value |
|--------|-------|
| **Records** | 7,043 customers |
| **Features (Raw)** | 21 columns |
| **Features (Processed)** | 30 features |
| **Target Variable** | Churn (Yes/No) |
| **Class Distribution** | 73.5% No Churn, 26.5% Churn |
| **Source** | Kaggle - https://www.kaggle.com/datasets/blastchar/telco-customer-churn |

### Feature Types
- **Demographics**: Gender, Senior Citizen, Partner, Dependents (4 features)
- **Service Usage**: Phone Service, Internet Service, Streaming TV/Movies (5 features)
- **Add-ons**: Online Security, Device Protection, Tech Support (3 features)
- **Billing**: Contract Type, Payment Method, Paperless Billing (3 features)
- **Metrics**: Tenure (months), Monthly Charges, Total Charges (3 features)
- **Encoded Features**: One-hot encoded categorical variables (7 features)

## Usage Guide

### Option 1: Data Pipeline (End-to-End)

```bash
# 1. Data Preprocessing
python << 'EOF'
from src.data_preprocessing import DataPreprocessor
preprocessor = DataPreprocessor()
df = preprocessor.load_data('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
X, y = preprocessor.prepare_features_and_target(df)
print(f"✅ Processed: {X.shape[0]} samples, {X.shape[1]} features")
EOF

# 2. Model Training
python << 'EOF'
from src.train_model import ModelTrainer
import pandas as pd
X = pd.read_csv('data/X_processed.csv')
y = pd.read_csv('data/y_processed.csv').values.ravel()
trainer = ModelTrainer()
model, metrics = trainer.train_logistic_regression(X, y)
print(f"✅ Accuracy: {metrics['accuracy']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
EOF

# 3. Make Predictions
python << 'EOF'
from src.predict import ChurnPredictor
import pandas as pd
predictor = ChurnPredictor()
predictor.load_model('models/best_model.pkl')
X_processed = pd.read_csv('data/X_processed.csv')
predictions = predictor.predict_batch(X_processed)
print(f"✅ Churn Rate: {(predictions['Predicted_Churn']==1).sum()/len(predictions)*100:.1f}%")
EOF
```

### Option 2: REST API (Real-time Predictions) ⭐ RECOMMENDED

**Start API Server:**
```bash
source venv/bin/activate
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Open Interactive API Documentation:**
```
http://localhost:8000/docs
```

**Or Test via cURL:**
```bash
# Health Check
curl http://localhost:8000/health

# Get Model Info
curl http://localhost:8000/model-info

# Get Features
curl http://localhost:8000/features

# Make Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Male","senior_citizen":0,"partner":"Yes",...}'

# Batch Prediction (all 7,043 customers)
curl -X POST http://localhost:8000/predict-batch
```

### Option 3: Testing

**Quick Test (5 minutes):**
```bash
source venv/bin/activate
cat TESTING_GUIDE.txt  # Follow Quick Test section
```

**Comprehensive Testing (20 minutes):**
```bash
# See all testing methods in TESTING_GUIDE.txt
# Includes: preprocessing, training, predictions, API, integration tests
```

**API Testing:**
```bash
cat API_TESTING_LOCAL.md  # Step-by-step API testing guide
```

## Model Performance

### Logistic Regression (Selected Model)
| Metric | Value |
|--------|-------|
| **Accuracy** | 80.34% |
| **Precision** | 65.2% |
| **Recall** | 55.61% |
| **F1-Score** | 60.03% |
| **ROC-AUC** | 0.8427 |
| **Training Samples** | 5,634 |
| **Test Samples** | 1,409 |

### Top 5 Churn Risk Factors
1. **Contract_Two year** (-1.35) - Strongest protection against churn
2. **tenure** (-1.26) - Each additional month reduces churn
3. **PhoneService** (-0.91) - Phone service customers are more loyal
4. **Contract_One year** (-0.70) - Some protection from 1-year contract
5. **OnlineSecurity_Yes** (-0.51) - Security add-on reduces churn

### Churn Rate Insights
- **By Contract Type**:
  - Month-to-month: 42.7% churn (HIGH RISK)
  - One year: 11.3% churn
  - Two year: 2.8% churn (LOWEST RISK)

- **By Internet Service**:
  - Fiber optic: 41.9% churn
  - DSL: 19.0% churn
  - No internet: 7.4% churn

- **By Tenure**:
  - Churned customers average: 18.0 months
  - Retained customers average: 37.6 months

## API Endpoints (8 Total)

### Information Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root endpoint |
| GET | `/health` | Health check - verify API is running |
| GET | `/model-info` | Model type, accuracy, ROC-AUC, precision, recall |
| GET | `/features` | List all 30 input features |
| GET | `/feature-importance` | Top 10 most influential features |
| GET | `/example-input` | Example customer data format |

### Prediction Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Single customer churn prediction with risk level |
| POST | `/predict-batch` | Batch predictions for all 7,043 customers |

### Response Format
```json
{
  "prediction": 1,
  "churn_probability": 0.75,
  "risk_level": "High",
  "top_risk_factors": [
    {"feature": "Contract_Two year", "contribution": -0.35},
    {"feature": "tenure", "contribution": -0.28}
  ]
}
```

### Risk Levels
- 🟢 **Low**: Churn probability ≤ 30%
- 🟡 **Medium**: Churn probability 30-60%
- 🔴 **High**: Churn probability > 60%
