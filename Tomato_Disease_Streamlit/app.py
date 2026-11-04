
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from PIL import Image


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Tomato Disease Recognition",
    page_icon="🍅",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR /
    "tomato_cnn_outputs" /
    "tomato_disease_cnn_final.keras"
)

MAPPING_PATH = (
    BASE_DIR /
    "tomato_cnn_outputs" /
    "class_mapping.json"
)

IMAGE_SIZE = (160, 160)


# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model_and_classes():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    with open(MAPPING_PATH, "r", encoding="utf-8") as file:
        mapping = json.load(file)

    classes = [
        mapping[key]
        for key in sorted(mapping.keys(), key=int)
    ]

    if model.output_shape[-1] != len(classes):
        raise ValueError(
            "Model output and class mapping do not match."
        )

    if tuple(model.input_shape[1:]) != (160, 160, 3):
        raise ValueError(
            f"Expected 160x160 RGB input, got {model.input_shape}"
        )

    return model, classes


# ============================================================
# 3. IMAGE PREPROCESSING
# ============================================================

def preprocess_image(uploaded_file):

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    resized_image = image.resize(
        IMAGE_SIZE
    )

    image_array = np.asarray(
        resized_image,
        dtype=np.float32
    )

    # The trained CNN already contains Rescaling(1./255).
    # Do not normalize the image a second time.

    image_batch = np.expand_dims(
        image_array,
        axis=0
    )

    return image, image_batch


# ============================================================
# 4. GENERATE CLASSIFICATION REPORT
# ============================================================

def generate_report(
    filename,
    predicted_class,
    confidence,
    probabilities,
    class_names
):

    timestamp = datetime.now().astimezone().strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )

    scores = pd.DataFrame({
        "Class": class_names,
        "Probability (%)": probabilities * 100
    })

    scores = scores.sort_values(
        "Probability (%)",
        ascending=False
    )

    report = [
        "TOMATO PLANT DISEASE CLASSIFICATION REPORT",
        "=" * 48,
        f"Image: {filename}",
        f"Date: {timestamp}",
        f"Predicted class: {predicted_class}",
        f"Top model score: {confidence:.2%}",
        "",
        "CLASS SCORES",
        "-" * 48
    ]

    for _, row in scores.iterrows():

        report.append(
            f"{row['Class']}: "
            f"{row['Probability (%)']:.2f}%"
        )

    report.extend([
        "",
        "INTERPRETATION",
        "-" * 48,
        "This is a CNN model prediction, not a verified diagnosis.",
        "Softmax scores are not necessarily calibrated probabilities.",
        "Image quality and dataset differences may affect predictions.",
        "Consult a qualified agricultural specialist for important decisions."
    ])

    return timestamp, scores, "\n".join(report)


# ============================================================
# 5. APPLICATION HEADER
# ============================================================

st.title(
    "🍅 Tomato Plant Disease Recognition"
)

st.write(
    "Upload a tomato leaf image to identify its "
    "predicted disease class using a trained CNN."
)

st.sidebar.title("Model Information")

st.sidebar.write("Architecture: CNN")
st.sidebar.write("Framework: TensorFlow / Keras")
st.sidebar.write("Image size: 160 × 160")
st.sidebar.write("Execution: CPU compatible")


# ============================================================
# 6. LOAD MODEL AND CLASS LABELS
# ============================================================

try:

    model, class_names = load_model_and_classes()

    st.sidebar.success(
        "Model loaded successfully"
    )

except Exception as error:

    st.error(
        f"Model loading failed: {error}"
    )

    st.stop()


# ============================================================
# 7. IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a tomato leaf image",
    type=["jpg", "jpeg", "png", "bmp"]
)


# ============================================================
# 8. IMAGE RECOGNITION
# ============================================================

if uploaded_file is not None:

    try:

        original_image, image_batch = preprocess_image(
            uploaded_file
        )

        with st.spinner("Analyzing image..."):

            predictions = model.predict(
                image_batch,
                verbose=0
            )[0]

        predicted_index = int(
            np.argmax(predictions)
        )

        predicted_class = class_names[
            predicted_index
        ]

        confidence = float(
            predictions[predicted_index]
        )

        # ----------------------------------------------------
        # DISPLAY IMAGE AND RESULT
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Uploaded Image")

            st.image(
                original_image,
                use_container_width=True
            )

        with col2:

            st.subheader("Recognition Result")

            st.metric(
                "Predicted Class",
                predicted_class
            )

            st.metric(
                "Top Model Score",
                f"{confidence:.2%}"
            )

            if confidence < 0.60:

                st.warning(
                    "The top score is relatively low. "
                    "Interpret this prediction cautiously."
                )

            else:

                st.info(
                    "This is the model's top prediction, "
                    "not a guarantee of correctness."
                )

        # ----------------------------------------------------
        # CLASS SCORES
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "Classification Scores"
        )

        timestamp, scores_df, report_text = generate_report(
            uploaded_file.name,
            predicted_class,
            confidence,
            predictions,
            class_names
        )

        st.bar_chart(
            scores_df.set_index("Class")
        )

        st.dataframe(
            scores_df,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # CLASSIFICATION REPORT
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "Classification Report"
        )

        st.caption(
            f"Report generated: {timestamp}"
        )

        st.text_area(
            "Report Preview",
            report_text,
            height=300
        )

        # ----------------------------------------------------
        # DOWNLOAD REPORT
        # ----------------------------------------------------

        filename = Path(
            uploaded_file.name
        ).stem.replace(" ", "_")

        st.download_button(
            label="Download Classification Report",
            data=report_text,
            file_name=f"{filename}_report.txt",
            mime="text/plain"
        )

        st.download_button(
            label="Download Class Scores CSV",
            data=scores_df.to_csv(index=False),
            file_name=f"{filename}_scores.csv",
            mime="text/csv"
        )

    except Exception as error:

        st.error(
            f"Image processing failed: {error}"
        )

else:

    st.info(
        "Upload an image to start disease recognition."
    )


# ============================================================
# 9. FOOTER
# ============================================================

st.divider()

st.caption(
    "Tomato Disease CNN Research Prototype | "
    "TensorFlow and Streamlit"
)

st.caption(
    "This application is intended for research and educational "
    "purposes. Validate the model using independent test data "
    "before real-world agricultural use."
)
