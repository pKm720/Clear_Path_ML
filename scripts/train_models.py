import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Configuration
FEATURES_PATH = "data/processed/training_features.csv"
MODELS_DIR = "models"

def train_virtual_sensor_models():
    """
    Execution of the training pipeline for virtual air quality sensors.
    Individual XGBoost models are generated for each suburban location.
    """
    if not os.path.exists(FEATURES_PATH):
        print(f"Error: {FEATURES_PATH} not found. Please run preprocessing first.")
        return

    print("Loading feature matrix...")
    df = pd.read_csv(FEATURES_PATH)
    
    # Identification of unique virtual sensor locations
    virtual_sensors = df['virtual_sensor'].unique()
    print(f"Detected {len(virtual_sensors)} virtual sensor locations for training.")

    # Preparation of directory for model persistence
    os.makedirs(MODELS_DIR, exist_ok=True)

    performance_summary = []

    for sensor in virtual_sensors:
        print(f"\n--- Training Model for: {sensor} ---")
        
        # Filtering data for the specific sensor location
        sensor_df = df[df['virtual_sensor'] == sensor]
        
        # Feature Selection
        # Temporal features + Distance/Value cues from nearest physical stations
        X = sensor_df[[
            'hour', 'day_of_week', 'is_rush_hour', 
            'dist_nearest_1', 'val_nearest_1', 
            'dist_nearest_2', 'val_nearest_2', 
            'dist_nearest_3', 'val_nearest_3'
        ]]
        y = sensor_df['target_idw']

        # Splitting data: 80% Training / 20% Evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Configuration of the XGBoost Regressor
        # Parameters optimized for tabular environmental data
        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42
        )

        # Training process
        model.fit(X_train, y_train)

        # Evaluation
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f"  Accuracy (R²): {r2:.4f}")
        print(f"  Mean Absolute Error: {mae:.4f} PM2.5 units")

        # Persistence of the trained brain
        model_filename = f"{sensor.lower().replace(' ', '_')}_xgboost.joblib"
        model_path = os.path.join(MODELS_DIR, model_filename)
        joblib.dump(model, model_path)
        print(f"  Model saved to: {model_path}")

        performance_summary.append({
            'sensor': sensor,
            'r2': r2,
            'mae': mae
        })

    # Final reporting
    print("\n" + "="*30)
    print("FINAL TRAINING REPORT")
    print("="*30)
    summary_df = pd.DataFrame(performance_summary)
    print(summary_df.to_string(index=False))
    
    avg_r2 = summary_df['r2'].mean()
    print(f"\nAverage City-Wide Accuracy (R²): {avg_r2:.4f}")

if __name__ == "__main__":
    train_virtual_sensor_models()
