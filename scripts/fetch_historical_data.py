import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time

# Load environment variables from .env if present
load_dotenv()

# Configuration
# This script uses the OpenAQ v3 API to fetch historical PM2.5 hourly averages for Bengaluru.
API_KEY = os.getenv('OPENAQ_API_KEY')
BASE_URL = "https://api.openaq.org/v3"
# Bengaluru Bounding Box: [minLon, minLat, maxLon, maxLat]
BBOX = "77.3,12.8,77.9,13.2" 
RAW_DATA_PATH = "data/raw/bangalore_aqi_historical.csv"

def fetch_locations():
    """Find all sensors within the Bengaluru bounding box using the v3 API."""
    print(f"Searching for sensors in BBOX: {BBOX}...")
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    
    params = {
        "bbox": BBOX,
        "limit": 100
    }
    
    response = requests.get(f"{BASE_URL}/locations", params=params, headers=headers)
    if response.status_code != 200:
        print(f"Error fetching locations: {response.text}")
        return []
        
    locations = response.json().get('results', [])
    print(f"Found {len(locations)} locations in the region.")
    return locations

def fetch_hourly_data(sensor_id, date_from, date_to):
    """Fetch hourly averages for a specific sensor ID."""
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    
    params = {
        "date_from": date_from,
        "date_to": date_to,
        "limit": 1000
    }
    
    endpoint = f"{BASE_URL}/sensors/{sensor_id}/hours"
    
    all_data = []
    page = 1
    
    while True:
        params['page'] = page
        response = requests.get(endpoint, params=params, headers=headers)
        
        if response.status_code != 200:
            print(f"    Error fetching hours for sensor {sensor_id}: {response.status_code}")
            break
            
        data = response.json().get('results', [])
        if not data:
            break
            
        all_data.extend(data)
        if len(data) < 1000:
            break
        page += 1
        time.sleep(0.5) # Compliance with rate limiting
        
    return all_data

def main():
    if not API_KEY:
        print("WARNING: OPENAQ_API_KEY not found in .env. API requests are rate-limited.")
        
    locations = fetch_locations()
    if not locations:
        return

    # Acquisition timeframe: Last 6 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    date_from = start_date.strftime("%Y-%m-%dT%H:%M:%S")
    date_to = end_date.strftime("%Y-%m-%dT%H:%M:%S")
    
    print(f"Fetching hourly data from {date_from} to {date_to}...")
    
    master_records = []
    
    for loc in locations:
        loc_id = loc['id']
        name = loc['name']
        coords = loc['coordinates']
        
        # Identify PM2.5 sensors within the location results
        pm25_sensors = [s for s in loc.get('sensors', []) if s.get('parameter', {}).get('name') == 'pm25']
        
        if not pm25_sensors:
            print(f"Processing Location: {name} - No PM2.5 sensor found. Skipping.")
            continue
            
        for sensor in pm25_sensors:
            sensor_id = sensor['id']
            print(f"Processing Sensor: {sensor_id} at {name}...")
            
            hours = fetch_hourly_data(sensor_id, date_from, date_to)
            print(f"  Fetched {len(hours)} hourly records.")
            
            for h in hours:
                master_records.append({
                    'location_id': loc_id,
                    'sensor_id': sensor_id,
                    'location_name': name,
                    'latitude': coords['latitude'],
                    'longitude': coords['longitude'],
                    'value': h['value'],
                    'timestamp': h['period']['datetimeFrom']
                })
            
    if master_records:
        df = pd.DataFrame(master_records)
        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
        df.to_csv(RAW_DATA_PATH, index=False)
        print(f"\nSUCCESS: Saved {len(master_records)} records to {RAW_DATA_PATH}")
    else:
        print("\nNo records found to save. Ensure the API key has the necessary permissions.")

if __name__ == "__main__":
    main()
