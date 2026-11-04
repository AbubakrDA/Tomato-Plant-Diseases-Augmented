"""Model loader and inference manager for tomato disease classification."""

import io
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

from api.metadata import DISEASE_METADATA

# Force CPU inference for predictable latency and portability
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], "GPU")

IMAGE_SIZE: Tuple[int, int] = (160, 160)


class TomatoClassifierService:
    def __init__(self, base_dir: Path = None):
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent.parent

        self.base_dir = base_dir
        self.model_path = self._locate_file("tomato_cnn_outputs/tomato_disease_cnn_final.keras")
        self.mapping_path = self._locate_file("tomato_cnn_outputs/class_mapping.json")
        self.metrics_path = self._locate_file("tomato_cnn_outputs/validation_metrics.json")

        self.model: tf.keras.Model = None
        self.class_names: List[str] = []
        self.class_mapping: Dict[str, str] = {}
        self.metrics: Dict = {}

    def _locate_file(self, relative_path: str) -> Path:
        """Find a file either in root/path or in subdirectories."""
        primary = self.base_dir / relative_path
        if primary.exists():
            return primary
        secondary = self.base_dir / "Tomato_Disease_Streamlit" / relative_path
        if secondary.exists():
            return secondary
        return primary

    def load_model(self) -> None:
        """Loads Keras model, class mapping, and validation metrics."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")
        if not self.mapping_path.exists():
            raise FileNotFoundError(f"Class mapping file not found at: {self.mapping_path}")

        # Load weights and architecture
        self.model = tf.keras.models.load_model(self.model_path)

        # Load class index mapping
        with open(self.mapping_path, "r", encoding="utf-8") as f:
            self.class_mapping = json.load(f)

        self.class_names = [
            self.class_mapping[str(i)]
            for i in range(len(self.class_mapping))
        ]

        # Load metrics if available
        if self.metrics_path.exists():
            with open(self.metrics_path, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Validates and prepares an uploaded image for CNN inference."""
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Invalid image file: {exc}")

        image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        img_array = np.asarray(image, dtype=np.float32)

        # Reshape to (1, 160, 160, 3) batch
        img_batch = np.expand_dims(img_array, axis=0)
        return img_batch

    def predict(self, image_bytes: bytes) -> Dict:
        """Performs classification on image bytes and enriches with metadata."""
        if not self.is_loaded:
            self.load_model()

        img_batch = self.preprocess_image(image_bytes)

        start_time = time.perf_counter()
        raw_preds = self.model.predict(img_batch, verbose=0)[0]
        inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        predicted_idx = int(np.argmax(raw_preds))
        predicted_class = self.class_names[predicted_idx]
        confidence = float(raw_preds[predicted_idx])

        # All classes probability dictionary
        all_probabilities = {
            cls_name: round(float(prob), 4)
            for cls_name, prob in zip(self.class_names, raw_preds)
        }

        # Retrieve domain metadata
        meta = DISEASE_METADATA.get(predicted_class, {
            "display_name": predicted_class.replace("Tomato___", "").replace("_", " "),
            "scientific_agent": "N/A",
            "is_healthy": "healthy" in predicted_class.lower(),
            "symptoms": "N/A",
            "treatment_recommendation": "Consult a qualified agricultural extension specialist."
        })

        return {
            "predicted_class": predicted_class,
            "display_name": meta["display_name"],
            "class_index": predicted_idx,
            "confidence": round(confidence, 4),
            "confidence_percentage": f"{confidence * 100:.2f}%",
            "is_healthy": meta["is_healthy"],
            "scientific_agent": meta.get("scientific_agent"),
            "symptoms": meta.get("symptoms"),
            "treatment_recommendation": meta.get("treatment_recommendation"),
            "inference_time_ms": inference_time_ms,
            "all_probabilities": all_probabilities
        }


# Global singleton instance
classifier = TomatoClassifierService()

