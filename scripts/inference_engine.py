import os
import json
import joblib
import pandas as pd
import numpy as np
import requests
import random
from datetime import datetime
from dotenv import load_dotenv
from math import radians, cos, sin, asin, sqrt

# Configuration
load_dotenv()
OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")
VIRTUAL_SENSORS_PATH = "config/virtual_sensors.json"
MODELS_DIR = "models"
BBOX = "77.3,12.8,77.9,13.2" # Bengaluru (minLon, minLat, maxLon, maxLat)

class InferenceEngine:
    """
    Singleton class to manage machine learning model loading and 
    real-time inference with automated scientific mock fallbacks.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InferenceEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.models = {}
        self.virtual_sensors = []
        self._load_resources()
        self._initialized = True

    def _load_resources(self):
        """
        Loading of virtual sensor configurations and serialized XGBoost models.
        """
        if os.path.exists(VIRTUAL_SENSORS_PATH):
            with open(VIRTUAL_SENSORS_PATH, 'r') as f:
                self.virtual_sensors = json.load(f)['virtual_sensors']
        
        for vs in self.virtual_sensors:
            name = vs['name']
            model_filename = f"{name.lower().replace(' ', '_')}_xgboost.joblib"
            model_path = os.path.join(MODELS_DIR, model_filename)
            
            if os.path.exists(model_path):
                self.models[name] = joblib.load(model_path)

    @staticmethod
    def haversine(lon1, lat1, lon2, lat2):
        """
        Great circle distance calculation using the Haversine formula.
        """
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 6371 # Earth radius
        return c * r

    def _generate_simulated_data(self):
        """
        Generation of scientific mock physical sensor data for Bengaluru.
        This provides a high-fidelity fallback when the OpenAQ API is unreachable.
        Estimates follow a diurnal traffic cycle (higher at 9 AM and 7 PM).
        """
        now = datetime.now()
        hour = now.hour
        # Base PM2.5 in Bengaluru: 35 units
        # Rush Hour Boost: +25 units
        is_rush = 1 if (8 <= hour <= 10 or 18 <= hour <= 20) else 0
        base_aqi = 35 + (is_rush * 25)
        
        # Representative physical sensor locations (Seed locations)
        mock_locations = [
            {"name": "BTM Layout", "lat": 12.9128, "lon": 77.6092},
            {"name": "Peenya", "lat": 13.0329, "lon": 77.5273},
            {"name": "Silk Board", "lat": 12.9172, "lon": 77.6228},
            {"name": "Indiranagar", "lat": 12.9719, "lon": 77.6412},
            {"name": "Church Street", "lat": 12.9744, "lon": 77.6015}
        ]
        
        simulated_data = []
        for loc in mock_locations:
            # Adding random variations for realistic modeling
            variation = random.uniform(-10, 15)
            simulated_data.append({
                "name": f"{loc['name']} (Simulated)",
                "lat": loc['lat'],
                "lon": loc['lon'],
                "value": round(max(5, base_aqi + variation), 2)
            })
        return simulated_data

    def _fetch_live_physical_data(self):
        """
        Retrieval of measurements using the /v3/locations geographic discovery endpoint.
        This provides live readings for all physical sensors in the Bengaluru bounding box.
        """
        # Utilizing the proven /locations endpoint from Phase 1 for live data connectivity
        url = f"https://api.openaq.org/v3/locations?bbox={BBOX}&limit=100"
        headers = {"X-API-Key": OPENAQ_API_KEY} if OPENAQ_API_KEY else {}
        
        try:
            # Low timeout for a responsive real-time user experience
            response = requests.get(url, headers=headers, timeout=3)
            response.raise_for_status()
            locations = response.json().get('results', [])
            
            live_data = []
            for loc in locations:
                coords = loc.get('coordinates', {})
                sensors = loc.get('sensors', [])
                
                # Iterate through sensors at this location to find PM2.5 data
                for s in sensors:
                    if s.get('parameter', {}).get('name') == 'pm25':
                        # The locations endpoint in v3 often includes a summary of the latest reading
                        latest_val = s.get('latest')
                        if latest_val is not None:
                            live_data.append({
                                "name": loc.get('name', 'Physical Sensor'),
                                "lat": coords.get('latitude'),
                                "lon": coords.get('longitude'),
                                "value": latest_val # Value is typically a floating point number here
                            })
            return live_data
        except Exception as e:
            print(f"Inference Engine: Live API connectivity error ({e}). Switching to Simulation Mode.")
            return []

    def get_predictions(self):
        """
        Execution of inference with automated scientific mock fallback.
        """
        physical_data = self._fetch_live_physical_data()
        data_source = "live"
        
        if not physical_data:
            physical_data = self._generate_simulated_data()
            data_source = "simulated_fallback"
            
        now = datetime.now()
        h, dow = now.hour, now.weekday()
        irh = 1 if (8 <= h <= 10 or 18 <= h <= 20) else 0
        
        predictions = []
        
        for vs in self.virtual_sensors:
            vs_name = vs['name']
            if vs_name not in self.models: continue
            
            vs_lat, vs_lon = vs['lat'], vs['lng']
            distances = []
            for s in physical_data:
                dist = self.haversine(vs_lon, vs_lat, s['lon'], s['lat'])
                distances.append((dist, s['value']))
            
            distances.sort(key=lambda x: x[0])
            
            # Pad nearest sensors if fewer than 3 found (from simulation or sparse API)
            if len(distances) >= 3:
                nearest = distances[:3]
            else:
                nearest = (distances * 3)[:3] 
                
            features = pd.DataFrame([{
                'hour': h, 'day_of_week': dow, 'is_rush_hour': irh,
                'dist_nearest_1': nearest[0][0], 'val_nearest_1': nearest[0][1],
                'dist_nearest_2': nearest[1][0], 'val_nearest_2': nearest[1][1],
                'dist_nearest_3': nearest[2][0], 'val_nearest_3': nearest[2][1]
            }])
            
            pred = self.models[vs_name].predict(features)[0]
            
            predictions.append({
                "location": vs_name,
                "lat": vs_lat, "lng": vs_lon,
                "predicted_pm25": round(max(0, float(pred)), 2),
                "timestamp": now.strftime("%Y-%m-%dT%H:%M:%S"),
                "status": "virtual_sensor",
                "data_source": data_source
            })
            
        return predictions

if __name__ == "__main__":
    engine = InferenceEngine()
    print(json.dumps(engine.get_predictions(), indent=2))
