#main.py
from fastapi import FastAPI, HTTPException, Depends, Query
from sqlalchemy.orm import Session
import joblib
import numpy as np
from typing import List, Optional
import json

from app import models, schemas, crud
from app.database import SessionLocal, engine, get_db

# Create database tables
models.Base.metadata.create_all(bind=engine)

# Load trained model
try:
    model_data = joblib.load("models/random_forest_model.pkl")
    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']
    model_metrics = model_data['metrics']
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Warning: Model file not found. Please train the model first.")
    model = None
    scaler = None
    features = None
    model_metrics = None

app = FastAPI(
    title="Mining Weather Prediction API",
    description="API untuk prediksi delay transportasi tambang berdasarkan kondisi cuaca",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {
        "message": "Mining Weather Prediction API",
        "status": "active",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "predict": "/predict",
            "batch-predict": "/predict/batch",
            "weather-data": "/weather-data",
            "model-metrics": "/model/metrics"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "database": "connected"
    }

@app.get("/model/metrics")
def get_model_metrics():
    if model_metrics is None:
        raise HTTPException(status_code=404, detail="Model metrics not available")
    
    return {
        "metrics": model_metrics,
        "features": features
    }

@app.post("/weather-data/", response_model=schemas.WeatherData)
def create_weather_data(
    weather_data: schemas.WeatherDataCreate,
    db: Session = Depends(get_db)
):
    """Create new weather data record"""
    return crud.create_weather_data(db, weather_data)

@app.get("/weather-data/", response_model=List[schemas.WeatherData])
def read_weather_data(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get all weather data"""
    return crud.get_all_weather_data(db, skip=skip, limit=limit)

@app.get("/weather-data/{weather_data_id}", response_model=schemas.WeatherData)
def read_weather_data_by_id(
    weather_data_id: int,
    db: Session = Depends(get_db)
):
    """Get weather data by ID"""
    db_weather_data = crud.get_weather_data(db, weather_data_id)
    if db_weather_data is None:
        raise HTTPException(status_code=404, detail="Weather data not found")
    return db_weather_data

@app.put("/weather-data/{weather_data_id}", response_model=schemas.WeatherData)
def update_weather_data(
    weather_data_id: int,
    weather_data_update: schemas.WeatherDataUpdate,
    db: Session = Depends(get_db)
):
    """Update weather data with actual delay hours"""
    db_weather_data = crud.update_weather_data(
        db, weather_data_id, weather_data_update
    )
    if db_weather_data is None:
        raise HTTPException(status_code=404, detail="Weather data not found")
    return db_weather_data

def prepare_features(weather_data: schemas.WeatherDataBase):
    """Prepare features for model prediction"""
    weather_severity, road_risk_score = crud.calculate_features(weather_data)
    
    feature_values = [
        weather_data.temperature,
        weather_data.humidity,
        weather_data.wind_speed,
        weather_data.rainfall,
        weather_data.visibility,
        weather_data.road_condition,
        weather_data.soil_moisture,
        weather_data.mining_activity_level,
        weather_severity,
        road_risk_score
    ]
    
    return np.array([feature_values]), weather_severity, road_risk_score

def get_recommendation(predicted_delay: float, high_risk: bool):
    """Get recommendation based on prediction"""
    if predicted_delay < 2:
        return "Transportasi dapat berjalan normal dengan pengawasan rutin."
    elif predicted_delay < 4:
        return "Waspada, persiapkan rencana kontingensi dan tingkatkan pengawasan."
    elif predicted_delay < 6:
        return "Pertimbangkan penundaan transportasi atau gunakan rute alternatif."
    else:
        return "Transportasi sebaiknya ditunda. Kondisi berisiko tinggi."

@app.post("/predict/", response_model=schemas.PredictionResponse)
def predict_delay(
    weather_data: schemas.WeatherDataCreate,
    db: Session = Depends(get_db)
):
    """Predict transport delay based on weather conditions"""
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Create weather data record
        db_weather_data = crud.create_weather_data(db, weather_data)
        
        # Prepare features for prediction
        features_array, weather_severity, road_risk_score = prepare_features(weather_data)
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        predicted_delay = float(model.predict(features_scaled)[0])
        high_delay_risk = predicted_delay > 4
        
        # Calculate confidence score (simulated)
        confidence_score = max(0.7, 1 - (predicted_delay * 0.05))
        confidence_score = min(confidence_score, 0.95)
        
        # Update weather data with prediction
        db_weather_data.predicted_delay_hours = predicted_delay
        db_weather_data.high_delay_risk = high_delay_risk
        db.commit()
        db.refresh(db_weather_data)
        
        # Create prediction record
        prediction_data = schemas.PredictionCreate(
            weather_data_id=db_weather_data.id,
            predicted_delay_hours=predicted_delay,
            high_delay_risk=high_delay_risk,
            confidence_score=confidence_score,
            features_used=json.dumps(features.tolist() if features else [])
        )
        
        db_prediction = crud.create_prediction(db, prediction_data)
        
        return schemas.PredictionResponse(
            prediction_id=db_prediction.id,
            weather_data_id=db_weather_data.id,
            predicted_delay_hours=predicted_delay,
            high_delay_risk=high_delay_risk,
            confidence_score=confidence_score,
            recommendation=get_recommendation(predicted_delay, high_delay_risk)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch/")
def batch_predict(
    batch_request: schemas.BatchPredictionRequest,
    db: Session = Depends(get_db)
):
    """Batch prediction for multiple weather data points"""
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    predictions = []
    
    for weather_data in batch_request.data:
        try:
            # Prepare features
            features_array, _, _ = prepare_features(weather_data)
            features_scaled = scaler.transform(features_array)
            
            # Make prediction
            predicted_delay = float(model.predict(features_scaled)[0])
            high_delay_risk = predicted_delay > 4
            
            predictions.append({
                "temperature": weather_data.temperature,
                "humidity": weather_data.humidity,
                "rainfall": weather_data.rainfall,
                "predicted_delay_hours": predicted_delay,
                "high_delay_risk": high_delay_risk,
                "recommendation": get_recommendation(predicted_delay, high_delay_risk)
            })
        except Exception as e:
            predictions.append({
                "error": str(e),
                "data": weather_data.dict()
            })
    
    return {
        "total_predictions": len(predictions),
        "successful": len([p for p in predictions if "error" not in p]),
        "predictions": predictions
    }

@app.get("/predictions/")
def get_predictions(
    weather_data_id: Optional[int] = None,
    limit: int = Query(50, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Get predictions with optional filtering"""
    if weather_data_id:
        predictions = crud.get_predictions_by_weather_data(db, weather_data_id)
    else:
        predictions = crud.get_recent_predictions(db, limit)
    
    return {
        "count": len(predictions),
        "predictions": predictions
    }

@app.get("/analytics/summary")
def get_analytics_summary(db: Session = Depends(get_db)):
    """Get analytics summary"""
    from sqlalchemy import func
    
    # Calculate statistics
    total_predictions = db.query(models.Prediction).count()
    high_risk_count = db.query(models.Prediction).filter(
        models.Prediction.high_delay_risk == True
    ).count()
    
    avg_delay = db.query(func.avg(models.Prediction.predicted_delay_hours)).scalar()
    max_delay = db.query(func.max(models.Prediction.predicted_delay_hours)).scalar()
    min_delay = db.query(func.min(models.Prediction.predicted_delay_hours)).scalar()
    
    return {
        "total_predictions": total_predictions,
        "high_risk_predictions": high_risk_count,
        "high_risk_percentage": (
            (high_risk_count / total_predictions * 100) if total_predictions > 0 else 0
        ),
        "average_predicted_delay": round(avg_delay or 0, 2),
        "maximum_predicted_delay": round(max_delay or 0, 2),
        "minimum_predicted_delay": round(min_delay or 0, 2)
    }