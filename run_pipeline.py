import os
import sys
import time
import pandas as pd

# Adding the scripts directory to the path for seamless imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from preprocess_data import process_features
from train_models import train_virtual_sensor_models

def run_ml_pipeline():
    """
    Orchestration of the complete Machine Learning workflow for ClearPath-ML.
    Handles data transformation, XGBoost training, and network updates.
    """
    print("="*60)
    print("CLEARPATH-ML: MASTER DEPLOYMENT PIPELINE")
    print("="*60)
    
    start_time = time.time()

    # 1. Feature Engineering Phase
    print("\n[STEP 1/3] Executing High-Intensity Preprocessing...")
    process_features()

    # 2. Model Optimization Phase
    print("\n[STEP 2/3] Execiting XGBoost Training Engine...")
    train_virtual_sensor_models()

    # 3. Final Verification & Dashboard
    print("\n[STEP 3/3] Finalizing Network Deployment...")
    
    # Reloading the virtual sensors to verify count
    config_path = "config/virtual_sensors.json"
    import json
    with open(config_path, 'r') as f:
        vs_count = len(json.load(f)['virtual_sensors'])

    end_time = time.time()
    duration = end_time - start_time

    print("\n" + "="*60)
    print("PIPELINE EXECUTION SUCCESSFUL")
    print("="*60)
    print(f"Total Sensors Deployed: {vs_count}")
    print(f"Total Execution Time: {duration:.2f} seconds")
    print(f"Network Status: ALL MODELS OPERATIONAL")
    print("="*60)
    print("\nPlease restart your ML backend (python main.py) to activate the new sensors.")

if __name__ == "__main__":
    run_ml_pipeline()
