from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
from predict import predict_data, get_wine_features_info


app = FastAPI(
    title="Wine Classification API",
    description="A FastAPI service for classifying wine types using machine learning",
    version="1.0.0"
)

class WineData(BaseModel):
    """Wine features for classification"""
    alcohol: float = Field(..., description="Alcohol percentage", ge=0, le=20)
    malic_acid: float = Field(..., description="Malic acid content", ge=0)
    ash: float = Field(..., description="Ash content", ge=0)
    alcalinity_of_ash: float = Field(..., description="Alcalinity of ash", ge=0)
    magnesium: float = Field(..., description="Magnesium content", ge=0)
    total_phenols: float = Field(..., description="Total phenols", ge=0)
    flavanoids: float = Field(..., description="Flavanoids content", ge=0)
    nonflavanoid_phenols: float = Field(..., description="Non-flavanoid phenols", ge=0)
    proanthocyanins: float = Field(..., description="Proanthocyanins content", ge=0)
    color_intensity: float = Field(..., description="Color intensity", ge=0)
    hue: float = Field(..., description="Hue value", ge=0)
    od280_od315_of_diluted_wines: float = Field(..., description="OD280/OD315 ratio of diluted wines", ge=0)
    proline: float = Field(..., description="Proline content", ge=0)

    class Config:
        schema_extra = {
            "example": {
                "alcohol": 14.23,
                "malic_acid": 1.71,
                "ash": 2.43,
                "alcalinity_of_ash": 15.6,
                "magnesium": 127.0,
                "total_phenols": 2.8,
                "flavanoids": 3.06,
                "nonflavanoid_phenols": 0.28,
                "proanthocyanins": 2.29,
                "color_intensity": 5.64,
                "hue": 1.04,
                "od280_od315_of_diluted_wines": 3.92,
                "proline": 1065.0
            }
        }

class WineResponse(BaseModel):
    """Response model for wine classification"""
    prediction: int = Field(..., description="Predicted wine class (0, 1, or 2)")
    class_name: str = Field(..., description="Name of the predicted wine class")
    confidence: float = Field(..., description="Prediction confidence score (0-1)")

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Wine Classification API",
        "version": "1.0.0"
    }

@app.get("/info", status_code=status.HTTP_200_OK)
async def get_info():
    """Get information about the wine classification service"""
    return {
        "service": "Wine Classification API",
        "description": "Classify wine types based on chemical analysis",
        "features": get_wine_features_info(),
        "classes": {
            0: "class_0",
            1: "class_1", 
            2: "class_2"
        },
        "model": "Random Forest Classifier",
        "features_count": 13
    }

@app.post("/predict", response_model=WineResponse)
async def predict_wine(wine_features: WineData):
    """
    Predict wine class based on chemical features
    
    - **Returns**: Wine class prediction with confidence score
    - **Classes**: 0 (class_0), 1 (class_1), 2 (class_2)
    """
    try:
        # Convert Pydantic model to feature array
        features = [[
            wine_features.alcohol,
            wine_features.malic_acid,
            wine_features.ash,
            wine_features.alcalinity_of_ash,
            wine_features.magnesium,
            wine_features.total_phenols,
            wine_features.flavanoids,
            wine_features.nonflavanoid_phenols,
            wine_features.proanthocyanins,
            wine_features.color_intensity,
            wine_features.hue,
            wine_features.od280_od315_of_diluted_wines,
            wine_features.proline
        ]]

        # Get prediction from the model
        prediction_result = predict_data(features)
        
        return WineResponse(
            prediction=prediction_result['prediction'],
            class_name=prediction_result['class_name'],
            confidence=prediction_result['confidence']
        )
    
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500, 
            detail="Model files not found. Please ensure the model is trained and saved."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
