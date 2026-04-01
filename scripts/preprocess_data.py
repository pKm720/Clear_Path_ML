import pandas as pd
import numpy as np
import os
import json
import ast
from datetime import datetime
from math import radians, cos, sin, asin, sqrt

# Configuration
RAW_DATA_PATH = "data/raw/bangalore_aqi_historical.csv"
VIRTUAL_SENSORS_PATH = "config/virtual_sensors.json"
PROCESSED_DATA_PATH = "data/processed/training_features.csv"
OUTLIER_THRESHOLD = 500

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculation of the great circle distance between two points 
    on the earth (specified in decimal degrees).
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    return c * r

def parse_openaq_timestamp(ts_str):
    """
    Extraction of the local ISO timestamp from the OpenAQ JSON-string format.
    Example: "{'utc': '...', 'local': '2024-03-20T10:00:00+05:30'}"
    """
    try:
        ts_dict = ast.literal_eval(ts_str)
        # Using tz_localize(None) ensures a clean index for pivoting
        return pd.to_datetime(ts_dict['local']).tz_localize(None)
    except:
        return None

def process_features():
    """
    Transformation of raw AQI data into a structured feature matrix.
    The implementation is optimized for large datasets using coordinate caching.
    """
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: {RAW_DATA_PATH} not found.")
        return

    print("Loading raw dataset...")
    df = pd.read_csv(RAW_DATA_PATH)
    
    # 1. Outlier Removal
    initial_count = len(df)
    df = df[df['value'] <= OUTLIER_THRESHOLD]
    print(f"Removed {initial_count - len(df)} outliers (> {OUTLIER_THRESHOLD}).")

    # 2. Temporal Feature Engineering
    print("Parsing timestamps and creating temporal features...")
    df['dt_local'] = df['timestamp'].apply(parse_openaq_timestamp)
    df = df.dropna(subset=['dt_local'])
    
    df['hour'] = df['dt_local'].dt.hour
    df['day_of_week'] = df['dt_local'].dt.dayofweek
    df['month'] = df['dt_local'].dt.month
    
    # Peak traffic hours in Bengaluru: 08:00-10:00 and 18:00-20:00
    df['is_rush_hour'] = df['hour'].apply(lambda x: 1 if (8 <= x <= 10 or 18 <= x <= 20) else 0)

    # 3. Load Virtual Sensor Targets
    with open(VIRTUAL_SENSORS_PATH, 'r') as f:
        virtual_sensors = json.load(f)['virtual_sensors']

    # 4. Spatial Feature Engineering & Target Generation (IDW)
    print("Computing spatial distance matrix and IDW targets...")
    
    # Pivot physical sensor data: Rows are Timestamps, Columns are Sensor Locations
    pivot_df = df.pivot_table(index='dt_local', columns='location_name', values='value')
    
    # Performance Optimization: Cache physical sensor coordinates in a dictionary 
    # to avoid expensive DataFrame filtering inside the time-series loop.
    sensor_meta = df[['location_name', 'latitude', 'longitude']].drop_duplicates()
    coords_cache = {
        row['location_name']: (row['latitude'], row['longitude']) 
        for _, row in sensor_meta.iterrows()
    }
    
    rows = []
    total_hours = len(pivot_df)
    
    for i, (dt, reading_row) in enumerate(pivot_df.iterrows()):
        if i % 1000 == 0:
            print(f"  Processed {i}/{total_hours} timestamps...")

        # Identification of active sensors for this specific hour
        available_sensors = reading_row.dropna()
        if len(available_sensors) < 3:
            continue
            
        h = dt.hour
        dow = dt.dayofweek
        irh = 1 if (8 <= h <= 10 or 18 <= h <= 20) else 0
        
        # Convert Series to dictionary for faster iteration
        hourly_data = available_sensors.to_dict()
        
        for vs in virtual_sensors:
            vs_lat, vs_lon, vs_name = vs['lat'], vs['lng'], vs['name']
            
            distances = []
            for s_name, val in hourly_data.items():
                s_lat, s_lon = coords_cache[s_name]
                dist = haversine(vs_lon, vs_lat, s_lon, s_lat)
                distances.append((dist, val))
            
            # Selection of the 3 nearest physical sensors for IDW calculation
            distances.sort(key=lambda x: x[0])
            nearest = distances[:3]
            
            # Weighted average based on inverse distance
            weights = [1.0 / (d[0] + 0.1) for d in nearest]
            idw_target = sum(d[1] * w for d, w in zip(nearest, weights)) / sum(weights)
            
            rows.append({
                'timestamp': dt,
                'virtual_sensor': vs_name,
                'hour': h,
                'day_of_week': dow,
                'is_rush_hour': irh,
                'dist_nearest_1': nearest[0][0],
                'val_nearest_1': nearest[0][1],
                'dist_nearest_2': nearest[1][0],
                'val_nearest_2': nearest[1][1],
                'dist_nearest_3': nearest[2][0],
                'val_nearest_3': nearest[2][1],
                'target_idw': idw_target
            })

    # 5. Data Finalization & Export
    if rows:
        features_df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
        features_df.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"SUCCESS: Exported {len(features_df)} training samples to {PROCESSED_DATA_PATH}")
    else:
        print("Error: No training samples were generated.")

if __name__ == "__main__":
    process_features()
