import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

class EmployeeAttritionModel:
    def __init__(self, model_type='random_forest'):
        self.classifier = None
        self.scaler = None
        self.label_encoders = {}
        self.model_type = model_type
        self.features = ['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 
                       'JobSatisfaction', 'WorkLifeBalance', 'OverTime', 'BusinessTravel',
                       'DistanceFromHome', 'JobLevel']
        self.categorical_columns = ['BusinessTravel', 'OverTime']
        
    def train_model(self, data_path='HR.csv'):
        # Load the dataset
        data_set = pd.read_csv(data_path)
        
        # Data Preprocessing
        # Convert categorical variables to numerical
        le = LabelEncoder()
        
        # Store label encoders for later use
        for column in ['Attrition', 'BusinessTravel', 'Department', 'EducationField', 
                      'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']:
            if column in data_set.columns:
                self.label_encoders[column] = LabelEncoder()
                data_set[column] = self.label_encoders[column].fit_transform(data_set[column])
        
        # Feature Selection
        X = data_set[self.features]
        y = data_set['Attrition']
        
        # Splitting the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        
        # Feature Scaling
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Training the model
        if self.model_type == 'random_forest':
            self.classifier = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=42)
        elif self.model_type == 'svm':
            self.classifier = SVC(probability=True, random_state=42)
        elif self.model_type == 'logistic_regression':
            self.classifier = LogisticRegression(random_state=42, max_iter=1000)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        self.classifier.fit(X_train, y_train)
        
        # Model Evaluation
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy, classification_report(y_test, y_pred), confusion_matrix(y_test, y_pred)
    
    def predict_attrition(self, employee_data):
        """
        Predict attrition for a single employee
        employee_data: dict with keys matching the features
        """
        if self.classifier is None or self.scaler is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Convert input to DataFrame
        df = pd.DataFrame([employee_data])
        
        # Encode categorical variables
        for column in self.categorical_columns:
            if column in df.columns:
                df[column] = self.label_encoders[column].transform(df[column])
        
        # Ensure correct feature order
        df = df[self.features]
        
        # Scale features
        scaled_data = self.scaler.transform(df)
        
        # Make prediction
        prediction = self.classifier.predict(scaled_data)[0]
        probability = self.classifier.predict_proba(scaled_data)[0]
        
        return {
            'prediction': 'Leaving' if prediction == 1 else 'Staying',
            'probability_leaving': probability[1],
            'probability_staying': probability[0],
            'risk_level': 'High' if probability[1] > 0.4 else 'Medium' if probability[1] > 0.15 else 'Low'
        }
    
    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        if self.classifier is None:
            raise ValueError("Model not trained yet.")
        
        # For SVM and Logistic Regression, use coefficients as importance
        if self.model_type == 'random_forest':
            importance_values = self.classifier.feature_importances_
        elif self.model_type in ['svm', 'logistic_regression']:
            # For linear models, use absolute coefficients
            if hasattr(self.classifier, 'coef_'):
                importance_values = np.abs(self.classifier.coef_[0])
            else:
                # For non-linear SVM, return equal importance
                importance_values = np.ones(len(self.features)) / len(self.features)
        else:
            importance_values = np.ones(len(self.features)) / len(self.features)
        
        importance_df = pd.DataFrame({
            'Feature': self.features,
            'Importance': importance_values
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def save_model(self, model_path='attrition_model.pkl'):
        """Save the trained model and preprocessing objects"""
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'features': self.features,
            'model_type': self.model_type
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, model_path='attrition_model.pkl'):
        """Load a trained model"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.features = model_data['features']
        self.model_type = model_data.get('model_type', 'random_forest')
