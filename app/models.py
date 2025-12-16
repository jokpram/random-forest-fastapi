#models.py
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.sql import func
from app.database import Base

class WeatherData(Base):
    __tablename__ = "weather_data"
    
    id = Column(Integer, primary_key=True, index=True)
    temperature = Column(Float)
    humidity = Column(Float)
    wind_speed = Column(Float)
    rainfall = Column(Float)
    visibility = Column(Float)
    road_condition = Column(Integer)
    soil_moisture = Column(Float)
    mining_activity_level = Column(Integer)
    weather_severity = Column(Float, nullable=True)
    road_risk_score = Column(Float, nullable=True)
    transport_delay_hours = Column(Float, nullable=True)
    predicted_delay_hours = Column(Float, nullable=True)
    high_delay_risk = Column(Boolean, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    weather_data_id = Column(Integer, index=True)
    predicted_delay_hours = Column(Float)
    high_delay_risk = Column(Boolean)
    confidence_score = Column(Float)
    features_used = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())