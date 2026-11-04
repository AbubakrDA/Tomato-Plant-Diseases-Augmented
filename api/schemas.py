"""Pydantic request and response schemas for the FastAPI service."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    model_loaded: bool = Field(..., example=True)
    input_shape: List[int] = Field(..., example=[160, 160, 3])
    num_classes: int = Field(..., example=10)
    timestamp: str = Field(...)


class ClassListResponse(BaseModel):
    total_classes: int = Field(..., example=10)
    classes: List[str] = Field(...)
    class_mapping: Dict[str, str] = Field(...)


class ModelInfoResponse(BaseModel):
    model_name: str = Field(..., example="Tomato Leaf Disease CNN")
    framework: str = Field(..., example="TensorFlow / Keras")
    input_shape: List[int] = Field(..., example=[160, 160, 3])
    num_classes: int = Field(..., example=10)
    validation_accuracy: float = Field(..., example=0.9457)
    validation_loss: float = Field(..., example=0.1637)
    training_epochs: int = Field(..., example=20)
    batch_size: int = Field(..., example=16)


class PredictionResponse(BaseModel):
    predicted_class: str = Field(..., example="Tomato___Early_blight")
    display_name: str = Field(..., example="Early Blight")
    class_index: int = Field(..., example=1)
    confidence: float = Field(..., example=0.9825, description="Probability between 0.0 and 1.0")
    confidence_percentage: str = Field(..., example="98.25%")
    is_healthy: bool = Field(..., example=False)
    scientific_agent: Optional[str] = Field(None, example="Alternaria solani")
    symptoms: Optional[str] = Field(None)
    treatment_recommendation: Optional[str] = Field(None)
    inference_time_ms: float = Field(..., example=24.5)
    all_probabilities: Dict[str, float] = Field(...)
    disclaimer: str = Field(
        default="This is an automated machine learning prediction for decision support, not an official diagnostic certificate."
    )

