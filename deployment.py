"""
Model Deployment Module
Handles model serving, API creation, and monitoring integration
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ModelServer:
    """
    Flask-based model server for patient readmission prediction
    Includes security features and monitoring capabilities
    """
    
    def __init__(self, model_path, preprocessor_path):
        self.app = Flask(__name__)
        self.model = self.load_model(model_path)
        self.preprocessor = self.load_model(preprocessor_path)
        self.setup_logging()
        self.setup_routes()
        
        # Monitoring
        self.prediction_log = []
        self.performance_metrics = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0
        }
    
    def load_model(self, path):
        """Load trained model or preprocessor"""
        try:
            model = joblib.load(path)
            print(f"Successfully loaded model from {path}")
            return model
        except Exception as e:
            print(f"Error loading model from {path}: {e}")
            return None
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('deployment.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_routes(self):
        """Define API routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'model_loaded': self.model is not None
            })
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            """Main prediction endpoint"""
            try:
                # Log request
                self.logger.info("Received prediction request")
                self.performance_metrics['total_predictions'] += 1
                
                # Get data from request
                data = request.get_json()
                
                # Validate input
                if not data or 'patient_data' not in data:
                    return jsonify({'error': 'No patient data provided'}), 400
                
                # Convert to DataFrame
                patient_df = pd.DataFrame([data['patient_data']])
                
                # Preprocess data
                processed_data = self.preprocess_data(patient_df)
                
                # Make prediction
                prediction, probability = self.make_prediction(processed_data)
                
                # Log successful prediction
                self.log_prediction(data['patient_data'], prediction, probability)
                self.performance_metrics['successful_predictions'] += 1
                
                # Return response
                response = {
                    'prediction': int(prediction),
                    'probability': float(probability),
                    'risk_level': self.get_risk_level(probability),
                    'timestamp': datetime.now().isoformat(),
                    'model_version': '1.0.0'
                }
                
                return jsonify(response)
                
            except Exception as e:
                self.logger.error(f"Prediction error: {e}")
                self.performance_metrics['failed_predictions'] += 1
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/batch_predict', methods=['POST'])
        def batch_predict():
            """Batch prediction endpoint"""
            try:
                data = request.get_json()
                
                if not data or 'patients' not in data:
                    return jsonify({'error': 'No patient data provided'}), 400
                
                patients_df = pd.DataFrame(data['patients'])
                processed_data = self.preprocess_data(patients_df)
                
                predictions = self.model.predict(processed_data)
                probabilities = self.model.predict_proba(processed_data)[:, 1]
                
                results = []
                for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                    results.append({
                        'patient_id': i,
                        'prediction': int(pred),
                        'probability': float(prob),
                        'risk_level': self.get_risk_level(prob)
                    })
                
                return jsonify({'predictions': results})
                
            except Exception as e:
                self.logger.error(f"Batch prediction error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/metrics', methods=['GET'])
        def get_metrics():
            """Get deployment metrics"""
            return jsonify(self.performance_metrics)
    
    def preprocess_data(self, data):
        """Preprocess incoming data using the saved preprocessor"""
        try:
            # Handle missing values (simple imputation for demo)
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())
            
            categorical_cols = data.select_dtypes(include=['object']).columns
            data[categorical_cols] = data[categorical_cols].fillna('Unknown')
            
            # Apply the saved preprocessor
            processed_data = self.preprocessor.transform(data)
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Preprocessing error: {e}")
            raise e
    
    def make_prediction(self, processed_data):
        """Make prediction using the trained model"""
        try:
            probability = self.model.predict_proba(processed_data)[0, 1]
            prediction = self.model.predict(processed_data)[0]
            return prediction, probability
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            raise e
    
    def get_risk_level(self, probability):
        """Convert probability to risk level"""
        if probability < 0.3:
            return 'Low'
        elif probability < 0.7:
            return 'Medium'
        else:
            return 'High'
    
    def log_prediction(self, patient_data, prediction, probability):
        """Log prediction for monitoring"""
        log_entry = {
            'timestamp': datetime.now(),
            'patient_data': patient_data,
            'prediction': prediction,
            'probability': probability,
            'risk_level': self.get_risk_level(probability)
        }
        self.prediction_log.append(log_entry)
        
        # Keep only last 1000 entries
        if len(self.prediction_log) > 1000:
            self.prediction_log = self.prediction_log[-1000:]
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask server"""
        self.logger.info(f"Starting model server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

# Example usage and testing
if __name__ == "__main__":
    # This would typically be run in production
    # For demo purposes, we'll create a simple test
    
    # Initialize server (in practice, models would be pre-trained)
    server = ModelServer(
        model_path='../models/readmission_model.pkl',
        preprocessor_path='../models/preprocessor.pkl'
    )
    
    # Example patient data for testing
    example_patient = {
        'age': 72,
        'length_of_stay': 8,
        'num_medications': 12,
        'num_lab_procedures': 45,
        'num_procedures': 2,
        'num_diagnoses': 8,
        'emergency_visits': 1,
        'prior_admissions': 2,
        'glucose_level': 145,
        'bmi': 32,
        'blood_pressure': 140,
        'gender': 'Male',
        'race': 'White',
        'insurance': 'Medicare',
        'primary_diagnosis': 'Diabetes'
    }
    
    print("Example prediction request:")
    print(f"Patient data: {example_patient}")
    
    # Note: In production, this would be called via HTTP requests
    # For demo, we're showing the structure
    
    print("\nTo run the server:")
    print("python deployment.py")
    print("\nThen send POST requests to http://localhost:5000/predict")
