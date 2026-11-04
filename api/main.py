"""FastAPI REST API for Tomato Plant Disease Recognition."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.model import classifier
from api.schemas import (
    ClassListResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tomato-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model weights at startup."""
    logger.info("Initializing tomato disease CNN model...")
    try:
        classifier.load_model()
        logger.info(f"Model loaded successfully with {len(classifier.class_names)} classes.")
    except Exception as exc:
        logger.error(f"Failed to pre-load model: {exc}")
    yield
    logger.info("Shutting down API service.")


app = FastAPI(
    title="Tomato Plant Disease Classification API",
    description=(
        "Production-ready deep learning REST API for automated diagnosis and "
        "classification of 10 tomato plant disease conditions using Convolutional Neural Networks (CNN)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for external frontends (Streamlit, React, Vue, Mobile)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/",
    tags=["Root"],
    summary="API Root and Documentation Link"
)
async def root():
    return {
        "service": "Tomato Plant Disease Classification API",
        "version": "1.0.0",
        "status": "online",
        "docs_url": "/docs",
        "endpoints": {
            "health": "/health",
            "classes": "/classes",
            "model_info": "/model-info",
            "predict": "/predict [POST]"
        }
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Monitoring"],
    summary="Health check endpoint"
)
async def health():
    if not classifier.is_loaded:
        try:
            classifier.load_model()
        except Exception as exc:
            logger.error(f"Failed to load model during healthcheck: {exc}")

    return HealthResponse(
        status="healthy" if classifier.is_loaded else "degraded",
        model_loaded=classifier.is_loaded,
        input_shape=[160, 160, 3],
        num_classes=len(classifier.class_names) if classifier.is_loaded else 0,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.get(
    "/classes",
    response_model=ClassListResponse,
    tags=["Model Info"],
    summary="Get all supported tomato disease classes"
)
async def get_classes():
    if not classifier.is_loaded:
        classifier.load_model()

    return ClassListResponse(
        total_classes=len(classifier.class_names),
        classes=classifier.class_names,
        class_mapping=classifier.class_mapping
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["Model Info"],
    summary="Get model architecture and training benchmarks"
)
async def get_model_info():
    if not classifier.is_loaded:
        classifier.load_model()

    metrics = classifier.metrics
    return ModelInfoResponse(
        model_name="Tomato Disease CNN (MobileNetV2 Transfer Learning)",
        framework="TensorFlow / Keras",
        input_shape=[160, 160, 3],
        num_classes=metrics.get("num_classes", len(classifier.class_names)),
        validation_accuracy=round(float(metrics.get("validation_accuracy", 0.9457)), 4),
        validation_loss=round(float(metrics.get("validation_loss", 0.1637)), 4),
        training_epochs=metrics.get("epochs_completed", 20),
        batch_size=metrics.get("batch_size", 16)
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Classify a tomato leaf image",
    status_code=status.HTTP_200_OK
)
async def predict_image(file: UploadFile = File(..., description="Uploaded image file (JPEG, PNG, BMP)")):
    # Validate mime type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file '{file.filename}' is not an image (Content-Type: {file.content_type})."
        )

    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file uploaded."
            )

        result = classifier.predict(contents)
        return PredictionResponse(**result)

    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve)
        )
    except Exception as exc:
        logger.exception("Inference error occurred:")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}"
        )