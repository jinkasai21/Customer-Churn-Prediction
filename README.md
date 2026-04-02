# Telco Customer Churn Prediction

A machine learning project to predict customer churn in the telecommunications industry using the Telco Customer Churn dataset.

## Project Structure

```
churn-project/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── notebooks/
│   └── (Jupyter notebooks for exploration and analysis)
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   └── utils/
│       └── __init__.py
│
├── models/
│   └── (Trained model files)
│
├── app/
│   └── main.py
│
├── requirements.txt
└── README.md
```

## Installation

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The project uses the Telco Customer Churn dataset from Kaggle:
- **Source**: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **File**: WA_Fn-UseC_-Telco-Customer-Churn.csv
- **Size**: ~1000 rows, 21 columns

## Usage

### Data Preprocessing
```python
python src/data_preprocessing.py
```

### Model Training
```python
python src/train_model.py
```

### Making Predictions
```python
python src/predict.py
```

### Running the App
```python
python app/main.py
```

## Technologies Used

- Python 3.9+
- Pandas (data manipulation)
- Scikit-learn (machine learning)
- Matplotlib & Seaborn (visualization)
- Kaggle API (data download)

## Author

jinkasai21

## License

This project is open source and available under the MIT License.
