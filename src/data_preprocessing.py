"""
Data Preprocessing Module for Telco Customer Churn Prediction
Handles data loading, cleaning, encoding, and scaling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import warnings

warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    A class to handle data preprocessing for the Telco Customer Churn dataset
    """
    
    def __init__(self, data_path):
        """
        Initialize the preprocessor with data path
        
        Args:
            data_path (str): Path to the CSV file
        """
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        self.categorical_columns = []
        self.numerical_columns = []
        
    def load_data(self):
        """Load the dataset from CSV file"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✓ Data loaded successfully. Shape: {self.df.shape}")
            print(f"  Columns: {list(self.df.columns)}")
            return self.df
        except FileNotFoundError:
            print(f"✗ Error: File not found at {self.data_path}")
            return None
    
    def explore_data(self):
        """Display basic information about the dataset"""
        if self.df is None:
            print("✗ No data loaded. Call load_data() first.")
            return
        
        print("\n=== Dataset Overview ===")
        print(f"Shape: {self.df.shape}")
        print(f"\nFirst few rows:\n{self.df.head()}")
        print(f"\nData types:\n{self.df.dtypes}")
        print(f"\nMissing values:\n{self.df.isnull().sum()}")
        print(f"\nBasic statistics:\n{self.df.describe()}")
    
    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        print("\n=== Handling Missing Values ===")
        missing_count = self.df.isnull().sum().sum()
        
        if missing_count == 0:
            print("✓ No missing values found")
            return
        
        print(f"Found {missing_count} missing values")
        
        # For numerical columns, fill with median
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].median(), inplace=True)
                print(f"  Filled {col} with median")
        
        # For categorical columns, fill with mode
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                print(f"  Filled {col} with mode")
        
        print(f"✓ Missing values handled. Remaining: {self.df.isnull().sum().sum()}")
    
    def identify_feature_types(self):
        """Identify and separate categorical and numerical features"""
        print("\n=== Feature Type Identification ===")
        
        # Separate numerical and categorical columns
        self.numerical_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target variable from categorical if present
        if 'Churn' in self.categorical_columns:
            self.categorical_columns.remove('Churn')
        
        print(f"Numerical columns ({len(self.numerical_columns)}): {self.numerical_columns}")
        print(f"Categorical columns ({len(self.categorical_columns)}): {self.categorical_columns}")
        
        return self.numerical_columns, self.categorical_columns
    
    def encode_categorical_features(self):
        """
        Convert categorical variables to numerical using:
        - LabelEncoder for binary/ordinal features
        - OneHotEncoder for nominal features
        """
        print("\n=== Encoding Categorical Features ===")
        
        # Binary encoding columns (use LabelEncoder)
        binary_columns = ['Churn', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                         'PhoneService', 'PaperlessBilling']
        
        # Nominal encoding columns (use OneHotEncoder)
        nominal_columns = ['InternetService', 'OnlineSecurity', 'OnlineBackup', 
                          'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                          'Contract', 'PaymentMethod', 'MultipleLines']
        
        # Apply LabelEncoder to binary columns
        label_encoders = {}
        for col in binary_columns:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                label_encoders[col] = le
                print(f"✓ LabelEncoded: {col}")
        
        # Apply OneHotEncoder to nominal columns
        for col in nominal_columns:
            if col in self.df.columns:
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                self.df = pd.concat([self.df.drop(col, axis=1), dummies], axis=1)
                print(f"✓ OneHotEncoded: {col}")
        
        # Convert TotalCharges to numeric (handle non-numeric values)
        if 'TotalCharges' in self.df.columns:
            self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
            self.df['TotalCharges'].fillna(self.df['TotalCharges'].median(), inplace=True)
            print(f"✓ Converted TotalCharges to numeric")
        
        print(f"✓ All categorical features encoded")
        return label_encoders
    
    def scale_numerical_features(self):
        """Apply StandardScaler to numerical features"""
        print("\n=== Scaling Numerical Features ===")
        
        if not self.numerical_columns:
            print("✗ No numerical columns identified. Call identify_feature_types() first.")
            return None
        
        # Scale numerical columns
        self.df[self.numerical_columns] = self.scaler.fit_transform(self.df[self.numerical_columns])
        print(f"✓ Scaled {len(self.numerical_columns)} numerical features")
        print(f"  Features: {self.numerical_columns}")
        
        return self.scaler
    
    def prepare_features_and_target(self):
        """Separate features (X) and target (y)"""
        print("\n=== Preparing Features and Target ===")
        
        if 'Churn' not in self.df.columns:
            print("✗ Error: 'Churn' column not found in dataset")
            return None, None
        
        # Separate target variable
        self.y = self.df['Churn']
        # Drop Churn and customerID columns
        self.X = self.df.drop(['Churn', 'customerID'], axis=1, errors='ignore')
        
        print(f"✓ Features shape: {self.X.shape}")
        print(f"✓ Target shape: {self.y.shape}")
        print(f"  Feature columns: {list(self.X.columns)}")
        print(f"  Target distribution:\n{self.y.value_counts()}")
        
        return self.X, self.y
    
    def preprocess(self):
        """
        Execute the complete preprocessing pipeline
        """
        print("=" * 60)
        print("STARTING DATA PREPROCESSING")
        print("=" * 60)
        
        # Step 1: Load data
        if self.load_data() is None:
            return None, None
        
        # Step 2: Explore data
        self.explore_data()
        
        # Step 3: Handle missing values
        self.handle_missing_values()
        
        # Step 4: Identify feature types
        self.identify_feature_types()
        
        # Step 5: Encode categorical features
        self.encode_categorical_features()
        
        # Step 6: Scale numerical features
        self.scale_numerical_features()
        
        # Step 7: Prepare features and target
        X, y = self.prepare_features_and_target()
        
        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        return X, y
    
    def save_processed_data(self, output_dir='data'):
        """Save processed features and target to CSV files"""
        print(f"\n=== Saving Processed Data ===")
        
        if self.X is None or self.y is None:
            print("✗ No processed data to save. Run preprocess() first.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save features
        X_path = os.path.join(output_dir, 'X_processed.csv')
        self.X.to_csv(X_path, index=False)
        print(f"✓ Features saved to {X_path}")
        
        # Save target
        y_path = os.path.join(output_dir, 'y_processed.csv')
        self.y.to_csv(y_path, index=False)
        print(f"✓ Target saved to {y_path}")


def main():
    """Main function to run data preprocessing"""
    
    # Define data path
    data_path = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(data_path)
    
    # Run full preprocessing pipeline
    X, y = preprocessor.preprocess()
    
    # Save processed data
    if X is not None and y is not None:
        preprocessor.save_processed_data()
        
        print("\n=== Data Summary ===")
        print(f"Total samples: {len(X)}")
        print(f"Total features: {X.shape[1]}")
        print(f"Churn distribution:\n{y.value_counts(normalize=True)}")


if __name__ == '__main__':
    main()
