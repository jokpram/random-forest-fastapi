#crud.py
from sqlalchemy.orm import Session
import app.models as models
import app.schemas as schemas
from typing import List, Optional
import numpy as np

def calculate_features(weather_data: schemas.WeatherDataBase):
    """Calculate derived features"""
    weather_severity = (
        weather_data.rainfall * 0.3 +
        weather_data.wind_speed * 0.2 +
        (100 - weather_data.visibility) * 0.25 +
        weather_data.humidity * 0.15 +
        abs(30 - weather_data.temperature) * 0.1
    )
    
    road_risk_score = (
        (10 - weather_data.road_condition) * 0.4 +
        weather_data.soil_moisture * 0.3 +
        weather_data.rainfall * 0.3
    )
    
    return weather_severity, road_risk_score

def create_weather_data(db: Session, weather_data: schemas.WeatherDataCreate):
    """Create new weather data record"""
    weather_severity, road_risk_score = calculate_features(weather_data)
    
    db_weather_data = models.WeatherData(
        **weather_data.dict(),
        weather_severity=weather_severity,
        road_risk_score=road_risk_score
    )
    
    db.add(db_weather_data)
    db.commit()
    db.refresh(db_weather_data)
    return db_weather_data

def get_weather_data(db: Session, weather_data_id: int):
    """Get weather data by ID"""
    return db.query(models.WeatherData).filter(
        models.WeatherData.id == weather_data_id
    ).first()

def get_all_weather_data(db: Session, skip: int = 0, limit: int = 100):
    """Get all weather data with pagination"""
    return db.query(models.WeatherData).offset(skip).limit(limit).all()

def update_weather_data(
    db: Session, 
    weather_data_id: int, 
    weather_data_update: schemas.WeatherDataUpdate
):
    """Update weather data with actual delay hours"""
    db_weather_data = get_weather_data(db, weather_data_id)
    if db_weather_data:
        for key, value in weather_data_update.dict().items():
            if value is not None:
                setattr(db_weather_data, key, value)
        
        # Update high_delay_risk based on actual delay
        if weather_data_update.transport_delay_hours is not None:
            db_weather_data.high_delay_risk = (
                weather_data_update.transport_delay_hours > 4
            )
        
        db.commit()
        db.refresh(db_weather_data)
    return db_weather_data

def create_prediction(db: Session, prediction: schemas.PredictionCreate):
    """Create prediction record"""
    db_prediction = models.Prediction(**prediction.dict())
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction

def get_predictions_by_weather_data(db: Session, weather_data_id: int):
    """Get predictions for specific weather data"""
    return db.query(models.Prediction).filter(
        models.Prediction.weather_data_id == weather_data_id
    ).all()

def get_recent_predictions(db: Session, limit: int = 50):
    """Get recent predictions"""
    return db.query(models.Prediction).order_by(
        models.Prediction.created_at.desc()
    ).limit(limit).all()