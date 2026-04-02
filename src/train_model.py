"""
Model Training Module for Telco Customer Churn Prediction
Trains multiple models and evaluates their performance
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except (ImportError, Exception) as e:
    XGB_AVAILABLE = False
    print(f"⚠ Warning: XGBoost not available. Error: {e}")


class ModelTrainer:
    """
    A class to train and evaluate multiple machine learning models
    """
    
    def __init__(self, X_path='data/X_processed.csv', y_path='data/y_processed.csv', test_size=0.2, random_state=42):
        """
        Initialize the model trainer
        
        Args:
            X_path (str): Path to features CSV
            y_path (str): Path to target CSV
            test_size (float): Test set size ratio
            random_state (int): Random state for reproducibility
        """
        self.X_path = X_path
        self.y_path = y_path
        self.test_size = test_size
        self.random_state = random_state
        
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.models = {}
        self.predictions = {}
        self.evaluations = {}
        self.best_model = None
        self.best_model_name = None
    
    def load_data(self):
        """Load preprocessed features and target"""
        print("\n=== Loading Preprocessed Data ===")
        try:
            self.X = pd.read_csv(self.X_path)
            self.y = pd.read_csv(self.y_path).iloc[:, 0]  # Get first column as Series
            
            print(f"✓ Features loaded: {self.X.shape}")
            print(f"✓ Target loaded: {self.y.shape}")
            print(f"  Feature columns: {list(self.X.columns)[:5]}... ({len(self.X.columns)} total)")
            
            return self.X, self.y
        except FileNotFoundError as e:
            print(f"✗ Error: {e}")
            return None, None
    
    def split_data(self):
        """Split data into training and testing sets"""
        print("\n=== Splitting Data ===")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=self.y
        )
        
        print(f"✓ Training set: {self.X_train.shape[0]} samples")
        print(f"✓ Testing set: {self.X_test.shape[0]} samples")
        print(f"  Train-Test Ratio: {len(self.X_train)}/{len(self.X_test)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_logistic_regression(self):
        """Train Logistic Regression model"""
        print("\n=== Training Logistic Regression ===")
        
        model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        model.fit(self.X_train, self.y_train)
        
        self.models['Logistic Regression'] = model
        print("✓ Logistic Regression model trained")
        
        return model
    
    def train_random_forest(self):
        """Train Random Forest model"""
        print("\n=== Training Random Forest ===")
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        model.fit(self.X_train, self.y_train)
        
        self.models['Random Forest'] = model
        print("✓ Random Forest model trained")
        print(f"  Trees: 100, Max Depth: 10")
        
        return model
    
    def train_xgboost(self):
        """Train XGBoost model"""
        if not XGB_AVAILABLE:
            print("\n⚠ XGBoost not available. Skipping...")
            return None
        
        print("\n=== Training XGBoost ===")
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(self.X_train, self.y_train)
        
        self.models['XGBoost'] = model
        print("✓ XGBoost model trained")
        print(f"  Trees: 100, Max Depth: 6, Learning Rate: 0.1")
        
        return model
    
    def evaluate_model(self, model_name, model):
        """
        Evaluate a model on test set
        
        Args:
            model_name (str): Name of the model
            model: Trained model object
        """
        # Make predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        self.predictions[model_name] = y_pred
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        # Store evaluation metrics
        self.evaluations[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': confusion_matrix(self.y_test, y_pred),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"\n{'='*60}")
        print(f"{model_name} - Performance Metrics")
        print(f"{'='*60}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print(f"\nConfusion Matrix:")
        print(self.evaluations[model_name]['confusion_matrix'])
        print(f"\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['No Churn', 'Churn']))
    
    def compare_models(self):
        """Compare all trained models"""
        print(f"\n{'='*60}")
        print("MODEL COMPARISON")
        print(f"{'='*60}")
        
        comparison_df = pd.DataFrame()
        for model_name, metrics in self.evaluations.items():
            comparison_df[model_name] = {
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics['roc_auc']
            }
        
        comparison_df = comparison_df.T
        print(comparison_df.to_string())
        
        # Find best model
        best_model_idx = comparison_df['ROC-AUC'].idxmax()
        self.best_model_name = best_model_idx
        self.best_model = self.models[best_model_idx]
        
        print(f"\n{'='*60}")
        print(f"🏆 BEST MODEL: {self.best_model_name}")
        print(f"{'='*60}")
        print(f"ROC-AUC Score: {comparison_df.loc[best_model_idx, 'ROC-AUC']:.4f}")
        
        return comparison_df
    
    def plot_confusion_matrices(self, save_path='models/confusion_matrices.png'):
        """Plot confusion matrices for all models"""
        print(f"\n=== Plotting Confusion Matrices ===")
        
        n_models = len(self.evaluations)
        fig, axes = plt.subplots(1, n_models, figsize=(15, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, metrics) in enumerate(self.evaluations.items()):
            cm = metrics['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{model_name}\n(Accuracy: {metrics["accuracy"]:.3f})')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrices saved to {save_path}")
        plt.close()
    
    def plot_model_comparison(self, save_path='models/model_comparison.png'):
        """Plot comparison of model metrics"""
        print(f"\n=== Plotting Model Comparison ===")
        
        comparison_df = pd.DataFrame()
        for model_name, metrics in self.evaluations.items():
            comparison_df[model_name] = {
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics['roc_auc']
            }
        
        comparison_df = comparison_df.T
        
        fig, ax = plt.subplots(figsize=(10, 6))
        comparison_df.plot(kind='bar', ax=ax)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.legend(loc='lower right')
        ax.set_ylim([0, 1.0])
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Model comparison plot saved to {save_path}")
        plt.close()
    
    def save_best_model(self, save_path='models/best_model.pkl'):
        """Save the best model to disk"""
        print(f"\n=== Saving Best Model ===")
        
        if self.best_model is None:
            print("✗ No best model selected. Run compare_models() first.")
            return
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        print(f"✓ Best model ({self.best_model_name}) saved to {save_path}")
    
    def save_model_metadata(self, save_path='models/model_metadata.txt'):
        """Save model metadata and evaluation results"""
        print(f"\n=== Saving Model Metadata ===")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("MODEL TRAINING SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Best Model: {self.best_model_name}\n\n")
            
            f.write("EVALUATION METRICS:\n")
            f.write("-"*60 + "\n")
            
            for model_name, metrics in self.evaluations.items():
                f.write(f"\n{model_name}:\n")
                f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall:    {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score:  {metrics['f1_score']:.4f}\n")
                f.write(f"  ROC-AUC:   {metrics['roc_auc']:.4f}\n")
        
        print(f"✓ Model metadata saved to {save_path}")
    
    def train_and_evaluate(self):
        """Execute the complete training pipeline"""
        print("\n" + "="*60)
        print("STARTING MODEL TRAINING")
        print("="*60)
        
        # Load data
        if self.load_data() is None:
            return
        
        # Split data
        self.split_data()
        
        # Train models
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_xgboost()
        
        # Evaluate models
        for model_name, model in self.models.items():
            self.evaluate_model(model_name, model)
        
        # Compare models
        self.compare_models()
        
        # Plot results
        self.plot_confusion_matrices()
        self.plot_model_comparison()
        
        # Save models
        self.save_best_model()
        self.save_model_metadata()
        
        print("\n" + "="*60)
        print("MODEL TRAINING COMPLETED")
        print("="*60)


def main():
    """Main function to run model training"""
    
    trainer = ModelTrainer(
        X_path='data/X_processed.csv',
        y_path='data/y_processed.csv',
        test_size=0.2,
        random_state=42
    )
    
    trainer.train_and_evaluate()


if __name__ == '__main__':
    main()
