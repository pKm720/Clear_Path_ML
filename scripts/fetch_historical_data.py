import os
import requests
import pandas as pd
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time

# Load environment variables from .env if present
load_dotenv()

# Configuration
API_KEY = os.getenv('OPENAQ_API_KEY')
BASE_URL = "https://api.openaq.org/v3"
# Bengaluru Bounding Box: [minLon, minLat, maxLon, maxLat]
BBOX = "77.3,12.8,77.9,13.2" 
RAW_DATA_PATH = "data/raw/bangalore_aqi_historical.csv"

def fetch_locations():
    """Find all sensors within the Bengaluru bounding box."""
    print(f"Searching for sensors in BBOX: {BBOX}...")
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    
    params = {
        "bbox": BBOX,
        "limit": 100,
        "parameters_id": 2 # id 2 is usually PM2.5 in OpenAQ v3
    }
    
    response = requests.get(f"{BASE_URL}/locations", params=params, headers=headers)
    if response.status_code != 200:
        print(f"Error fetching locations: {response.text}")
        return []
        
    locations = response.json().get('results', [])
    print(f"Found {len(locations)} locations with PM2.5 data.")
    return locations

def fetch_measurements(location_id, date_from, date_to):
    """Fetch hourly measurements for a specific location and timeframe."""
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    
    # We'll use the 'sensors' endpoint if we want specific parameter IDs
    # In OpenAQ v3, locations have multiple sensors.
    params = {
        "date_from": date_from,
        "date_to": date_to,
        "limit": 1000,
        "parameters_id": 2 # PM2.5
    }
    
    # OpenAQ v3 structure: /locations/{id}/sensors might be needed for specific stats
    # But /measurements with location_id filter is standard.
    endpoint = f"{BASE_URL}/locations/{location_id}/measurements"
    
    all_data = []
    page = 1
    
    while True:
        params['page'] = page
        response = requests.get(endpoint, params=params, headers=headers)
        
        if response.status_code != 200:
            print(f"  Error fetching measurements for {location_id}: {response.text}")
            break
            
        data = response.json().get('results', [])
        if not data:
            break
            
        all_data.extend(data)
        if len(data) < 1000:
            break
        page += 1
        time.sleep(0.5) # Rate limiting friendly
        
    return all_data

def main():
    if not API_KEY:
        print("WARNING: OPENAQ_API_KEY not found in .env. Requests will be rate-limited.")
        
    locations = fetch_locations()
    if not locations:
        return

    # Set timeframe: Last 6 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    date_from = start_date.strftime("%Y-%m-%dT%H:%M:%S")
    date_to = end_date.strftime("%Y-%m-%dT%H:%M:%S")
    
    print(f"Fetching data from {date_from} to {date_to}...")
    
    master_records = []
    
    for loc in locations:
        loc_id = loc['id']
        name = loc['name']
        coords = loc['coordinates']
        print(f"Processing Location: {name} (ID: {loc_id})...")
        
        measurements = fetch_measurements(loc_id, date_from, date_to)
        print(f"  Fetched {len(measurements)} measurements.")
        
        for m in measurements:
            master_records.append({
                'location_id': loc_id,
                'location_name': name,
                'latitude': coords['latitude'],
                'longitude': coords['longitude'],
                'parameter': m['parameter']['name'],
                'value': m['value'],
                'unit': m['parameter']['units'],
                'timestamp': m['period']['datetimeFrom']
            })
            
    if master_records:
        df = pd.DataFrame(master_records)
        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
        df.to_csv(RAW_DATA_PATH, index=False)
        print(f"\nSUCCESS: Saved {len(master_records)} records to {RAW_DATA_PATH}")
    else:
        print("\nNo records found to save.")

if __name__ == "__main__":
    main()
