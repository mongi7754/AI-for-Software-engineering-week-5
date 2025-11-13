"""
Data Preprocessing Module for Patient Readmission Prediction
Handles data cleaning, feature engineering, and bias mitigation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    A comprehensive data preprocessing class for healthcare data
    Handles missing values, feature engineering, and bias mitigation
    """
    
    def __init__(self):
        self.preprocessor = None
        self.feature_names = None
        self.scaler = StandardScaler()
        
    def generate_synthetic_data(self, n_samples=1000):
        """
        Generate synthetic patient data for demonstration
        In real scenario, this would be replaced with actual EHR data
        """
        np.random.seed(42)
        
        data = {
            'age': np.random.normal(65, 15, n_samples),
            'length_of_stay': np.random.gamma(5, 2, n_samples),
            'num_medications': np.random.poisson(8, n_samples),
            'num_lab_procedures': np.random.poisson(45, n_samples),
            'num_procedures': np.random.poisson(1.5, n_samples),
            'num_diagnoses': np.random.poisson(7, n_samples),
            'emergency_visits': np.random.poisson(0.5, n_samples),
            'prior_admissions': np.random.poisson(1, n_samples),
            'glucose_level': np.random.normal(120, 30, n_samples),
            'bmi': np.random.normal(28, 6, n_samples),
            'blood_pressure': np.random.normal(130, 20, n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'race': np.random.choice(['White', 'Black', 'Hispanic', 'Asian'], n_samples, p=[0.6, 0.2, 0.15, 0.05]),
            'insurance': np.random.choice(['Private', 'Medicare', 'Medicaid', 'None'], n_samples),
            'primary_diagnosis': np.random.choice(['Diabetes', 'Heart_Failure', 'COPD', 'Pneumonia', 'Other'], n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Ensure realistic values
        df['age'] = df['age'].clip(18, 100)
        df['bmi'] = df['bmi'].clip(15, 50)
        df['glucose_level'] = df['glucose_level'].clip(70, 300)
        
        # Create target variable (readmission within 30 days)
        # Simulate realistic relationships with features
        readmission_prob = (
            0.1 + 
            0.0005 * (df['age'] - 65)**2 +
            0.02 * df['num_medications'] +
            0.01 * df['num_diagnoses'] +
            0.03 * df['prior_admissions'] -
            0.005 * df['glucose_level'].clip(70, 200) +
            (df['primary_diagnosis'] == 'Diabetes') * 0.1 +
            (df['primary_diagnosis'] == 'Heart_Failure') * 0.15
        )
        
        df['readmitted_30_days'] = np.random.binomial(1, readmission_prob.clip(0, 0.8))
        
        return df
    
    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset
        Uses different strategies for numerical and categorical features
        """
        print("Handling missing values...")
        
        # Numerical features - impute with median
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'readmitted_30_days' in numerical_features:
            numerical_features.remove('readmitted_30_days')
            
        numerical_imputer = SimpleImputer(strategy='median')
        df[numerical_features] = numerical_imputer.fit_transform(df[numerical_features])
        
        # Categorical features - impute with mode
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_features] = categorical_imputer.fit_transform(df[categorical_features])
        
        return df
    
    def feature_engineering(self, df):
        """
        Create new features based on domain knowledge
        Feature engineering is crucial for healthcare predictive models
        """
        print("Performing feature engineering...")
        
        # Create comorbidity index (simplified)
        df['comorbidity_index'] = (
            df['num_diagnoses'] * 0.1 + 
            df['num_medications'] * 0.05 +
            (df['primary_diagnosis'] == 'Diabetes') * 1 +
            (df['primary_diagnosis'] == 'Heart_Failure') * 2 +
            (df['primary_diagnosis'] == 'COPD') * 1.5
        )
        
        # Age categories
        df['age_group'] = pd.cut(df['age'], bins=[0, 45, 65, 75, 100], 
                                labels=['Young', 'Middle', 'Senior', 'Elderly'])
        
        # High-risk medication flag
        df['high_risk_meds'] = (df['num_medications'] > 10).astype(int)
        
        # Complex patient flag
        df['complex_patient'] = (
            (df['num_diagnoses'] > 5) & 
            (df['num_medications'] > 8) & 
            (df['prior_admissions'] > 1)
        ).astype(int)
        
        # Length of stay categories
        df['los_category'] = pd.cut(df['length_of_stay'], bins=[0, 3, 7, 14, 100], 
                                   labels=['Short', 'Medium', 'Long', 'Very_Long'])
        
        return df
    
    def mitigate_bias(self, df, sensitive_attribute='race'):
        """
        Basic bias mitigation through reweighting
        In practice, more advanced techniques like adversarial debiasing would be used
        """
        print("Applying bias mitigation...")
        
        # Calculate weights to balance representation
        weights = {}
        total_samples = len(df)
        
        for group in df[sensitive_attribute].unique():
            group_count = len(df[df[sensitive_attribute] == group])
            weights[group] = total_samples / (len(df[sensitive_attribute].unique()) * group_count)
        
        df['sample_weight'] = df[sensitive_attribute].map(weights)
        
        return df
    
    def prepare_features(self, df):
        """
        Prepare features for model training
        Includes encoding, scaling, and final preprocessing
        """
        print("Preparing features for modeling...")
        
        # Separate features and target
        X = df.drop('readmitted_30_days', axis=1)
        y = df['readmitted_30_days']
        
        # Define feature sets
        numerical_features = ['age', 'length_of_stay', 'num_medications', 'num_lab_procedures',
                            'num_procedures', 'num_diagnoses', 'emergency_visits', 'prior_admissions',
                            'glucose_level', 'bmi', 'blood_pressure', 'comorbidity_index']
        
        categorical_features = ['gender', 'race', 'insurance', 'primary_diagnosis', 
                               'age_group', 'los_category']
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', sparse=False), categorical_features)
            ])
        
        # Fit and transform the data
        X_processed = preprocessor.fit_transform(X)
        self.preprocessor = preprocessor
        self.feature_names = self._get_feature_names(preprocessor)
        
        return X_processed, y, preprocessor
    
    def _get_feature_names(self, preprocessor):
        """Extract feature names after preprocessing"""
        feature_names = []
        
        for name, transformer, features in preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(features)
            elif name == 'cat':
                # Get feature names from one-hot encoder
                cat_features = transformer.get_feature_names_out(features)
                feature_names.extend(cat_features)
        
        return feature_names
    
    def handle_class_imbalance(self, X, y):
        """
        Address class imbalance using SMOTE
        Important for healthcare where positive cases are often rare
        """
        print("Addressing class imbalance with SMOTE...")
        
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        print(f"Class distribution before SMOTE: {pd.Series(y).value_counts().to_dict()}")
        print(f"Class distribution after SMOTE: {pd.Series(y_balanced).value_counts().to_dict()}")
        
        return X_balanced, y_balanced

# Example usage
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    
    # Generate synthetic data
    df = preprocessor.generate_synthetic_data(1000)
    print("Synthetic data generated successfully!")
    print(f"Dataset shape: {df.shape}")
    print(f"Readmission rate: {df['readmitted_30_days'].mean():.2%}")
    
    # Demo preprocessing pipeline
    df_clean = preprocessor.handle_missing_values(df)
    df_engineered = preprocessor.feature_engineering(df_clean)
    df_balanced = preprocessor.mitigate_bias(df_engineered)
    
    X, y, preprocessor = preprocessor.prepare_features(df_balanced)
    print(f"Processed features shape: {X.shape}")
