"""Unit tests for the Tomato Disease Classifier and API logic."""

import io
from PIL import Image
import pytest

from api.model import TomatoClassifierService
from api.metadata import DISEASE_METADATA


@pytest.fixture(scope="module")
def classifier():
    service = TomatoClassifierService()
    service.load_model()
    return service


def create_test_image_bytes(format="JPEG") -> bytes:
    """Creates in-memory RGB dummy leaf image."""
    img = Image.new("RGB", (160, 160), color=(34, 139, 34))
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


def test_model_loading(classifier):
    """Verifies Keras model architecture and 10 disease classes are initialized."""
    assert classifier.is_loaded is True
    assert len(classifier.class_names) == 10
    assert len(classifier.class_mapping) == 10


def test_classes_integrity(classifier):
    """Ensures healthy and pathogenic classes are properly mapped."""
    assert "Tomato___healthy" in classifier.class_names
    assert "Tomato___Early_blight" in classifier.class_names
    assert "Tomato___Late_blight" in classifier.class_names
    assert "Tomato___Bacterial_spot" in classifier.class_names


def test_predict_pipeline(classifier):
    """Tests full preprocessing and inference pipeline."""
    img_bytes = create_test_image_bytes(format="JPEG")
    result = classifier.predict(img_bytes)

    assert result["predicted_class"] in classifier.class_names
    assert 0 <= result["class_index"] < 10
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["confidence_percentage"], str)
    assert "%" in result["confidence_percentage"]
    assert "display_name" in result
    assert "treatment_recommendation" in result
    assert "inference_time_ms" in result
    assert result["inference_time_ms"] > 0
    assert len(result["all_probabilities"]) == 10


def test_invalid_image_bytes(classifier):
    """Verifies ValueError is raised when corrupted image is supplied."""
    with pytest.raises(ValueError):
        classifier.preprocess_image(b"not an image file")


def test_disease_metadata_coverage():
    """Validates that all 10 classes have complete agronomic metadata."""
    assert len(DISEASE_METADATA) == 10
    for key, data in DISEASE_METADATA.items():
        assert "display_name" in data
        assert "symptoms" in data
        assert "treatment_recommendation" in data
        assert isinstance(data["is_healthy"], bool)