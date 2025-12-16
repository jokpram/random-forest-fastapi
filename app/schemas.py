#schemas.py
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class WeatherDataBase(BaseModel):
    temperature: float
    humidity: float
    wind_speed: float
    rainfall: float
    visibility: float
    road_condition: int
    soil_moisture: float
    mining_activity_level: int

class WeatherDataCreate(WeatherDataBase):
    pass

class WeatherDataUpdate(BaseModel):
    transport_delay_hours: Optional[float] = None

class WeatherData(WeatherDataBase):
    id: int
    weather_severity: Optional[float] = None
    road_risk_score: Optional[float] = None
    transport_delay_hours: Optional[float] = None
    predicted_delay_hours: Optional[float] = None
    high_delay_risk: Optional[bool] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

class PredictionBase(BaseModel):
    weather_data_id: int
    predicted_delay_hours: float
    high_delay_risk: bool
    confidence_score: float
    features_used: str

class PredictionCreate(PredictionBase):
    pass

class Prediction(PredictionBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class PredictionResponse(BaseModel):
    prediction_id: int
    weather_data_id: int
    predicted_delay_hours: float
    high_delay_risk: bool
    confidence_score: float
    recommendation: str

class BatchPredictionRequest(BaseModel):
    data: List[WeatherDataBase]

class ModelMetrics(BaseModel):
    mae: float
    mse: float
    rmse: float
    r2: float