"""
Unit tests for data preprocessing module
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import DataPreprocessor

class TestDataPreprocessing(unittest.TestCase):
    
    def setUp(self):
        self.preprocessor = DataPreprocessor()
        self.sample_data = self.preprocessor.generate_synthetic_data(100)
    
    def test_data_generation(self):
        """Test that synthetic data is generated correctly"""
        self.assertIsInstance(self.sample_data, pd.DataFrame)
        self.assertGreater(len(self.sample_data), 0)
        self.assertIn('readmitted_30_days', self.sample_data.columns)
    
    def test_missing_value_handling(self):
        """Test missing value imputation"""
        # Introduce some missing values
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0:4, 'age'] = np.nan
        data_with_missing.loc[5:9, 'gender'] = None
        
        cleaned_data = self.preprocessor.handle_missing_values(data_with_missing)
        
        # Check that missing values are handled
        self.assertFalse(cleaned_data['age'].isna().any())
        self.assertFalse(cleaned_data['gender'].isna().any())
    
    def test_feature_engineering(self):
        """Test feature engineering creates new features"""
        engineered_data = self.preprocessor.feature_engineering(self.sample_data)
        
        # Check that new features are created
        self.assertIn('comorbidity_index', engineered_data.columns)
        self.assertIn('age_group', engineered_data.columns)
        self.assertIn('high_risk_meds', engineered_data.columns)
    
    def test_bias_mitigation(self):
        """Test bias mitigation adds sample weights"""
        mitigated_data = self.preprocessor.mitigate_bias(self.sample_data)
        self.assertIn('sample_weight', mitigated_data.columns)

if __name__ == '__main__':
    unittest.main()
