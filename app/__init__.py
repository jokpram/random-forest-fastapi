#__init__.py
"""
Mining Weather Prediction System

Aplikasi untuk memprediksi delay transportasi tambang berdasarkan kondisi cuaca
menggunakan model Random Forest dengan FastAPI dan PostgreSQL.
"""

__version__ = "1.0.0"
__author__ = "Mining Weather Prediction Team"
__description__ = "Sistem Prediksi Delay Transportasi Tambang Berbasis Cuaca"

# Import utama untuk kemudahan akses
from app.database import SessionLocal, engine, get_db
from app.models import WeatherData, Prediction
from app.schemas import (
    WeatherDataBase, WeatherDataCreate, WeatherData, 
    PredictionBase, PredictionCreate, Prediction, PredictionResponse
)
from app.crud import (
    create_weather_data, get_weather_data, get_all_weather_data,
    update_weather_data, create_prediction, get_predictions_by_weather_data,
    get_recent_predictions, calculate_features
)

# List module exports
__all__ = [
    # Versi dan metadata
    "__version__", "__author__", "__description__",
    
    # Database
    "SessionLocal", "engine", "get_db",
    
    # Models
    "WeatherData", "Prediction",
    
    # Schemas
    "WeatherDataBase", "WeatherDataCreate", "WeatherData",
    "PredictionBase", "PredictionCreate", "Prediction", "PredictionResponse",
    
    # CRUD operations
    "create_weather_data", "get_weather_data", "get_all_weather_data",
    "update_weather_data", "create_prediction", "get_predictions_by_weather_data",
    "get_recent_predictions", "calculate_features",
]

print(f"Mining Weather Prediction System v{__version__} initialized")