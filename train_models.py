import numpy as np
import pandas as pd
from model_utils import EmployeeAttritionModel

def train_all_models():
    """Train and save all three models: Random Forest, SVM, and Logistic Regression"""
    
    models = ['random_forest', 'svm', 'logistic_regression']
    results = {}
    
    for model_type in models:
        print(f"\n=== Training {model_type.replace('_', ' ').title()} Model ===")
        
        # Create model instance
        model = EmployeeAttritionModel(model_type)
        
        # Train the model
        accuracy, report, cm = model.train_model()
        
        # Save the model
        model_path = f'attrition_model_{model_type}.pkl'
        model.save_model(model_path)
        
        # Store results
        results[model_type] = {
            'accuracy': accuracy,
            'model_path': model_path
        }
        
        print(f"Model: {model_type.replace('_', ' ').title()}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Model saved to: {model_path}")
        print(f"Confusion Matrix:\n{cm}")
        print(f"Classification Report:\n{report}")
        print("-" * 50)
    
    # Summary
    print("\n=== Training Summary ===")
    for model_type, result in results.items():
        print(f"{model_type.replace('_', ' ').title()}: {result['accuracy']:.4f}")
    
    return results

if __name__ == "__main__":
    train_all_models()
