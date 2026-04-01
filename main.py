import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from scripts.inference_engine import InferenceEngine

# Initialization of the ClearPath-ML API
app = FastAPI(
    title="ClearPath-ML: Virtual Sensor Network",
    description="Real-time air quality inference engine for Bengaluru's geographic data gaps.",
    version="1.0.0"
)

# Configuration of Cross-Origin Resource Sharing (CORS)
# This allows the ClearPath web frontend to fetch live ML predictions.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, this should be restricted to the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton of the ML Inference Engine
# This pre-loads the 7 XGBoost models in memory during startup.
engine = InferenceEngine()

from typing import List, Optional
from pydantic import BaseModel

class SensorContext(BaseModel):
    name: Optional[str] = None
    lat: float
    lon: float
    value: float

class PredictRequest(BaseModel):
    physical_sensors: Optional[List[SensorContext]] = None

@app.get("/")
async def root():
    """
    Status check endpoint for the ML service.
    """
    return {
        "status": "online",
        "models_loaded": len(engine.models),
        "virtual_sensors_monitored": len(engine.virtual_sensors)
    }

@app.post("/predict")
async def get_realtime_predictions(request: Optional[PredictRequest] = None):
    """
    Primary endpoint for real-time air quality inference.
    Supports a 'Push' model where physical sensor context is provided in the body
    to break circular dependencies during system startup.
    """
    try:
        # Context extraction from the request body if present
        context = None
        if request and request.physical_sensors:
            context = [s.dict() for s in request.physical_sensors]
            
        predictions = engine.get_predictions(physical_context=context)
        
        if isinstance(predictions, dict) and "error" in predictions:
            raise HTTPException(status_code=503, detail=predictions["error"])
            
        return {
            "timestamp": predictions[0]["timestamp"] if predictions else None,
            "data": predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Launching the server on port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
