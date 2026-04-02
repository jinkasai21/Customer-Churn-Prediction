"""
Prediction Module for Telco Customer Churn Prediction
Uses the trained model to make predictions on new data
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings('ignore')


class ChurnPredictor:
    """
    A class to make churn predictions using the trained model
    """
    
    def __init__(self, model_path='models/best_model.pkl', scaler_path=None):
        """
        Initialize the predictor
        
        Args:
            model_path (str): Path to the saved model
            scaler_path (str): Path to the saved scaler (optional)
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model from disk"""
        print("\n=== Loading Model ===")
        
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✓ Model loaded successfully from {self.model_path}")
            return self.model
        except FileNotFoundError:
            print(f"✗ Error: Model file not found at {self.model_path}")
            return None
    
    def load_preprocessing_info(self):
        """Load preprocessing information from training data"""
        print("\n=== Loading Preprocessing Information ===")
        
        try:
            # Load processed features to get column names
            X_processed = pd.read_csv('data/X_processed.csv')
            self.feature_columns = X_processed.columns.tolist()
            
            # Fit scaler on processed data for reference
            self.scaler.fit(X_processed[['SeniorCitizen', 'tenure', 'MonthlyCharges']])
            
            print(f"✓ Loaded {len(self.feature_columns)} feature columns")
            return self.feature_columns
        except FileNotFoundError:
            print("✗ Error: Processed data not found. Run preprocessing first.")
            return None
    
    def predict_single_customer(self, customer_data):
        """
        Predict churn for a single customer
        
        Args:
            customer_data (dict): Dictionary containing customer features
            
        Returns:
            tuple: (prediction, probability)
        """
        if self.model is None:
            print("✗ Model not loaded")
            return None, None
        
        # Convert dict to DataFrame
        df = pd.DataFrame([customer_data])
        
        # Ensure all required columns are present
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Select only required columns in correct order
        df = df[self.feature_columns]
        
        # Make prediction
        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0]
        
        return prediction, probability
    
    def predict_batch(self, input_file='data/X_processed.csv', output_file='data/predictions.csv'):
        """
        Make predictions for a batch of customers
        
        Args:
            input_file (str): Path to input data CSV
            output_file (str): Path to save predictions
        """
        print(f"\n=== Batch Prediction ===")
        
        try:
            # Load data
            X = pd.read_csv(input_file)
            print(f"✓ Loaded {len(X)} samples for prediction")
            
            # Make predictions
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            
            # Create results dataframe
            results_df = pd.DataFrame({
                'Prediction': predictions,
                'Probability_No_Churn': probabilities[:, 0],
                'Probability_Churn': probabilities[:, 1]
            })
            
            # Add interpretation
            results_df['Prediction_Label'] = results_df['Prediction'].map({0: 'No Churn', 1: 'Churn'})
            results_df['Risk_Level'] = pd.cut(
                results_df['Probability_Churn'],
                bins=[0, 0.3, 0.6, 1.0],
                labels=['Low', 'Medium', 'High']
            )
            
            # Save results
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            results_df.to_csv(output_file, index=False)
            print(f"✓ Predictions saved to {output_file}")
            
            # Print summary
            print(f"\nPrediction Summary:")
            print(f"  Total predictions: {len(results_df)}")
            print(f"  Predicted Churn: {(predictions == 1).sum()} ({(predictions == 1).sum()/len(predictions)*100:.1f}%)")
            print(f"  Predicted No Churn: {(predictions == 0).sum()} ({(predictions == 0).sum()/len(predictions)*100:.1f}%)")
            print(f"\nRisk Level Distribution:")
            print(results_df['Risk_Level'].value_counts().sort_index())
            
            return results_df
        except FileNotFoundError:
            print(f"✗ Error: Input file not found at {input_file}")
            return None
    
    def predict_from_raw_data(self, raw_data_dict):
        """
        Predict churn from raw (unprocessed) customer data
        This is a template - would need full preprocessing logic
        
        Args:
            raw_data_dict (dict): Raw customer features
            
        Returns:
            tuple: (prediction, probability, interpretation)
        """
        print("\n=== Predicting from Raw Data ===")
        
        # Example raw data structure
        example = {
            'gender': 'Male',
            'SeniorCitizen': 0,
            'Partner': 'Yes',
            'Dependents': 'No',
            'tenure': 12,
            'PhoneService': 'Yes',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 65.0,
            'TotalCharges': 780.0
        }
        
        print("Note: Raw data prediction requires full preprocessing pipeline.")
        print("Using example data structure:")
        print(example)
        return None
    
    def get_feature_importance(self):
        """Get feature importance from the model (if available)"""
        print("\n=== Feature Importance ===")
        
        if not hasattr(self.model, 'coef_'):
            print("✗ Feature importance not available for this model type")
            return None
        
        if self.feature_columns is None:
            self.load_preprocessing_info()
        
        # Get coefficients
        coefficients = self.model.coef_[0]
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'Feature': self.feature_columns,
            'Coefficient': coefficients,
            'Abs_Coefficient': np.abs(coefficients)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))
        
        return importance_df
    
    def explain_prediction(self, customer_data):
        """
        Provide explanation for a prediction
        
        Args:
            customer_data (dict): Customer features
        """
        print("\n=== Prediction Explanation ===")
        
        prediction, probability = self.predict_single_customer(customer_data)
        
        if prediction is None:
            return
        
        print(f"\nCustomer Data:")
        for key, value in customer_data.items():
            print(f"  {key}: {value}")
        
        print(f"\nPrediction Result:")
        print(f"  Prediction: {'Churn' if prediction == 1 else 'No Churn'}")
        print(f"  Probability (No Churn): {probability[0]:.4f}")
        print(f"  Probability (Churn): {probability[1]:.4f}")
        
        if probability[1] > 0.6:
            print(f"\n⚠ HIGH RISK: This customer is likely to churn")
        elif probability[1] > 0.3:
            print(f"\n⚡ MEDIUM RISK: Monitor this customer")
        else:
            print(f"\n✓ LOW RISK: Customer is likely to stay")


def interactive_prediction():
    """Interactive prediction mode"""
    print("\n" + "="*60)
    print("INTERACTIVE CHURN PREDICTION")
    print("="*60)
    
    predictor = ChurnPredictor()
    predictor.load_preprocessing_info()
    
    if predictor.model is None:
        return
    
    # Get feature importance
    predictor.get_feature_importance()
    
    print("\n" + "="*60)
    print("Creating sample predictions...")
    print("="*60)
    
    # Create sample customer data
    sample_customer = {
        'gender': 1,  # Encoded
        'SeniorCitizen': -0.44,  # Scaled
        'Partner': 1,  # Encoded
        'Dependents': 0,  # Encoded
        'tenure': 0.93,  # Scaled
        'PhoneService': 1,  # Encoded
        'MultipleLines': 1,  # Encoded
        'PaperlessBilling': 1,  # Encoded
        'MonthlyCharges': 0.0,  # Scaled
        'TotalCharges': -1.0,  # Scaled
        'InternetService_Fiber optic': False,
        'InternetService_No': True,
        'OnlineSecurity_No internet service': True,
        'OnlineSecurity_Yes': False,
        'OnlineBackup_No internet service': True,
        'OnlineBackup_Yes': False,
        'DeviceProtection_No internet service': True,
        'DeviceProtection_Yes': False,
        'TechSupport_No internet service': True,
        'TechSupport_Yes': False,
        'StreamingTV_No internet service': True,
        'StreamingTV_Yes': False,
        'StreamingMovies_No internet service': True,
        'StreamingMovies_Yes': False,
        'Contract_One year': False,
        'Contract_Two year': False,
        'PaymentMethod_Credit card (automatic)': False,
        'PaymentMethod_Electronic check': False,
        'PaymentMethod_Mailed check': False,
    }
    
    predictor.explain_prediction(sample_customer)


def main():
    """Main function to run predictions"""
    
    print("="*60)
    print("CUSTOMER CHURN PREDICTION")
    print("="*60)
    
    # Initialize predictor
    predictor = ChurnPredictor(model_path='models/best_model.pkl')
    
    # Load preprocessing information
    predictor.load_preprocessing_info()
    
    if predictor.model is None:
        print("✗ Failed to load model. Exiting.")
        return
    
    # Get feature importance
    importance_df = predictor.get_feature_importance()
    
    # Make batch predictions
    results_df = predictor.predict_batch(
        input_file='data/X_processed.csv',
        output_file='data/predictions.csv'
    )
    
    # Interactive prediction
    interactive_prediction()
    
    print("\n" + "="*60)
    print("PREDICTION COMPLETED")
    print("="*60)


if __name__ == '__main__':
    main()
