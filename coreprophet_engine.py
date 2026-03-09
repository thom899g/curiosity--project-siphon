"""
Predictive Engine for forecasting bridge flows and gas differentials.
Architectural Choices:
- RandomForest for interpretability and handling non-linear patterns
- Feature scaling for consistent model performance
- Joblib for model persistence (handles large numpy arrays efficiently)
- Firebase integration for real-time feature updates
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import firebase_admin
from firebase_admin import firestore

logger = logging.getLogger(__name__)

class ProphetEngine:
    """Predictive engine for bridge flow and gas price forecasting"""
    
    def __init__(self, firestore_client):
        self.db = firestore_client
        self.model: Optional[RandomForestRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_columns: List[str] = []
        self._initialize_engine()
        
    def _initialize_engine(self) -> None:
        """Initialize the engine with model loading or training"""
        try:
            # Attempt to load existing model
            self.model = joblib.load('models/prophet_model.joblib')
            self.scaler = joblib.load('models/prophet_scaler.joblib')
            logger.info("Loaded existing Prophet model from disk")
            
            # Load feature columns from metadata
            with open('models/feature_columns.txt', 'r') as f:
                self.feature_columns = [line.strip() for line in f.readlines()]
                
        except (FileNotFoundError, EOFError) as e:
            logger.warning(f"Could not load model: {e}. Training new model.")
            self._train_initial_model()
        except Exception as e:
            logger.error(f"Unexpected error loading model: {e}")
            raise
    
    def _generate_initial_dataset(self) -> pd.DataFrame:
        """Generate synthetic training data when no historical data exists"""
        logger.info("Generating initial synthetic training dataset")
        
        # Create realistic time-series data
        np.random.seed(42)
        n_samples = 10000
        
        # Base features
        data = {
            'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='T'),
            'hour_of_day': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'eth_price_change': np.random.normal(0, 0.02, n_samples),  # 2% daily volatility
            'base_gas_prev_block': np.random.uniform(10, 100, n_samples),  # 10-100 gwei
            'l1_gas_forecast': np.random.uniform(20, 200, n_samples),  # 20-200 gwei
            'bridge_queue_size': np.random.poisson(5, n_samples)  # Poisson for queue modeling
        }
        
        # Target: bridge volume in next 5 minutes (ETH)
        # Realistic patterns: higher during US business hours, weekends lower
        base_pattern = (
            np.sin(data['hour_of_day'] / 24 * 2 * np.pi) * 0.5 +  # Daily cycle
            (data['day_of_week'] < 5).astype(int) * 0.3 +  # Weekday boost
            np.random.normal(1, 0.2, n_samples)  # Noise
        )
        
        data['bridge_volume_next_5min'] = np.clip(
            base_pattern * np.random.exponential(5, n_samples),
            0.1, 50  # 0.1 to 50 ETH
        )
        
        df = pd.DataFrame(data)
        
        # Save to Firebase for future reference
        try:
            batch = self.db.batch()
            collection_ref = self.db.collection('training_data')
            
            # Store in chunks to avoid Firebase limits
            for i, row in df.head(1000).iterrows():  # Store first 1000 samples
                doc_ref = collection_ref.document()
                batch.set(doc_ref, row.to_dict())
                
            batch.commit()
            logger.info("Stored initial training data in Firebase")
        except Exception as e:
            logger.warning(f"Could not store training data in Firebase: {e}")
        
        return df
    
    def _train_initial_model(self) -> None:
        """Train initial model with synthetic or loaded data"""
        logger.info("Training initial Prophet model")
        
        # Try to load existing data first
        try:
            df = pd.read_csv('data/historical_bridge_data.csv')
            logger.info("Loaded historical data from CSV")
        except FileNotFoundError:
            try:
                # Try to load from Firebase
                docs = self.db.collection('bridge_transactions').limit(1000).stream()
                data = []
                for doc in docs:
                    data.append(doc.to_dict())
                
                if data:
                    df = pd.DataFrame(data)
                    logger.info(f"Loaded {len(df)} records from Firebase")
                else:
                    # Generate synthetic data
                    df = self._generate_initial_dataset()
                    logger.info("Generated synthetic training data")
            except Exception as e:
                logger.error(f"Could not load data from Firebase: {e}")
                df = self._generate_initial_dataset()
        
        # Feature engineering
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
        df['gas_ratio'] = df['l1_gas_forecast'] / (df['base_gas_prev_block'] + 1)
        
        # Define features
        self.feature_columns = [
            'hour_sin', 'hour_cos',
            'day_of_week',
            'eth_price_change',
            'base_gas_prev_block',
            'l1_gas_forecast',
            'gas_ratio',
            'bridge_queue_size'
        ]
        
        # Ensure all features exist
        missing_features = [f for f in self.feature_columns if f not in df.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}. Using available features.")
            self.feature_columns = [f for f in self.feature_columns if f in df.columns]
        
        X = df[self.feature_columns].fillna(0)
        y = df['bridge_volume_next_5min'].fillna(0)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1  # Use all cores
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        X_test_scaled = self.scaler.transform(X_test)
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        logger.info(f"Model trained. Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
        
        # Save model and metadata
        import os
        os.makedirs('models', exist_ok=True)
        
        joblib.dump(self.model, 'models/prophet_model.joblib')
        joblib.dump(self.scaler, 'models/prophet_scaler.joblib')
        
        with open('models/feature_columns.txt', 'w') as f:
            for feature in self.feature_columns:
                f.write(f"{feature}\n")
        
        logger.info("Model saved to disk")
    
    def predict_bridge_flow(self, current_features: Dict[str, float]) -> Tuple[float, float]:
        """
        Predict bridge flow for next 5 minutes
        
        Args:
            current_features: Dictionary of current feature values
            
        Returns:
            Tuple of (predicted_volume_eth, confidence_interval)
            
        Edge Cases Handled:
            - Missing features (uses default values)
            - Model not initialized (returns safe default)
            - Invalid feature values (clamps to reasonable ranges)
        """
        if not self.model or not self.scaler:
            logger.error("Model not initialized. Returning safe default.")
            return 0.0, 1.0
        
        try:
            # Prepare feature vector
            feature_vector = []
            for feature in self.feature_columns:
                if feature in current_features:
                    # Clamp to reasonable values
                    if 'gas' in feature:
                        val = max(0, min(1000, current_features[feature]))
                    elif 'ratio' in feature:
                        val = max(0, min(100, current_features[feature]))
                    else:
                        val = current_features[feature]
                    feature_vector.append(val)
                else:
                    # Use median from training
                    logger.warning(f"Missing feature {feature}, using default 0")
                    feature_vector.append(0.0)
            
            # Scale and predict
            scaled_features = self.scaler.transform([feature_vector])
            prediction = self.model.predict(scaled_features)[0]
            
            # Get prediction confidence from tree variance
            tree_preds = [tree.predict(scaled_features)[0] for tree in self.model.estimators_]
            confidence = 1.0 / (np.std(tree_preds) + 1e-6)  # Inverse of variance
            
            # Ensure non-negative prediction
            prediction = max(0, prediction)
            
            logger.debug(f"Prediction: {prediction:.2f} ETH, Confidence: {confidence:.2f}")
            return prediction, min(1.0, confidence / 100)  # Normalize confidence
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0.0, 0.0
    
    def update_model(self, new_data: pd.DataFrame) -> None:
        """Update model with new data (online learning)"""