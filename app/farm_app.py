import streamlit as st
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import tempfile
import time

from detector import (
    load_model,
    detect_image,
    draw_detections
)

from recommender import (
    load_recommendations,
    calculate_cost
)

from predictor import (
    estimate_severity,
    predict_progression,
    update_progression_log
)

from chatbot import chatbot_response

from utils import (
    MODEL_PATH,
    CSV_PATH,
    LOG_PATH,
    OUTPUT_IMAGE_DIR,
    OUTPUT_VIDEO_DIR,
    OUTPUT_REPORT_DIR,
    SAMPLE_IMAGE_DIR,
    SAMPLE_VIDEO_DIR
)
# ==========================================================
# Streamlit Setup
# ==========================================================

st.set_page_config(
    page_title="Crop Disease AI",
    layout="wide"
)

st.title("🌾 AI Crop Disease Detection System")

st.markdown(
    "Upload crop images or videos to detect diseases and get fertilizer recommendations."
)
# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.header("⚙️ Settings")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.1,
    0.95,
    0.45,
    0.01
)

show_overlay = st.sidebar.checkbox(
    "Show Overlay Labels",
    value=True
)


# ==========================================================
# Load Model and Recommendations
# ==========================================================

@st.cache_resource
def cached_model():
    return load_model(MODEL_PATH)


@st.cache_data
def cached_recommendations():
    return load_recommendations(CSV_PATH)


model = cached_model()
recommendations = cached_recommendations()
# ==========================================================
# Session State
# ==========================================================

if "severity_history" not in st.session_state:
    st.session_state["severity_history"] = {}


# ==========================================================
# Tabs
# ==========================================================

image_tab, video_tab, chatbot_tab = st.tabs(
    [
        "🖼️ Image Detection",
        "🎥 Video Detection",
        "🤖 Farmer Chatbot"
    ]
)


# ==========================================================
# IMAGE TAB
# ==========================================================
with image_tab:

    st.header("🖼️ Upload Image")

    uploaded_image = st.file_uploader(
        "Upload a crop image",
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )

    if uploaded_image:

        pil_image = Image.open(uploaded_image).convert("RGB")

        image_array = np.array(pil_image)

        image_bgr = image_array[:, :, ::-1].copy()

        st.image(pil_image, caption="Uploaded Image", width=500)

        if st.button("Run Detection"):

            with st.spinner("Running Detection..."):
                result = detect_image(
                    model,
                    image_bgr,
                    confidence_threshold
                )

            if len(result.boxes) == 0:
                st.warning("No disease detected.")

            else:

                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()

                output_image = draw_detections(
                    image_bgr.copy(),
                    boxes,
                    classes,
                    confidences,
                    model.names,
                    recommendations,
                    show_overlay
                )

                st.image(
                    output_image[:, :, ::-1],
                    caption="Detection Result",
                    width=600
                )

                st.subheader("🩺 Disease Summary")

                detected = set()

                for cls in classes:

                    disease = model.names[int(cls)]

                    if disease in detected:
                        continue

                    detected.add(disease)

                    severity = estimate_severity(result)

                    st.session_state["severity_history"].setdefault(
                        disease,
                        []
                    ).append(severity)

                    progression = predict_progression(
                        st.session_state["severity_history"][disease]
                    )

                    update_progression_log(
                        LOG_PATH,
                        disease,
                        severity
                    )

                    st.markdown(f"### 🌿 {disease}")

                    st.write(f"Severity: {severity}%")

                    if progression is not None:
                        st.write(f"Predicted Future Severity: {progression}%")
                    else:
                        st.write("Predicted Future Severity: Insufficient data")

                    if disease in recommendations:

                        recommendation = recommendations[disease]

                        cost = calculate_cost(recommendation)

                        st.write(
                            f"Fertilizer: {recommendation['fertilizer_name']}"
                        )

                        st.write(
                            f"Type: {recommendation['type']}"
                        )

                        st.write(
                            f"Dosage: {recommendation['dosage']}"
                        )

                        st.write(
                            f"Brand: {recommendation['brand']}"
                        )

                        st.write(
                            f"Estimated Cost: ₹{cost}"
                        )
                    else:
                        st.info("No fertilizer recommendation available for this disease.")

                    st.markdown("---")

                report_path = OUTPUT_REPORT_DIR / "latest_report.txt"

                with open(report_path, "w", encoding="utf-8") as report:

                    report.write(
                        "Crop Disease Detection Report\n"
                    )

                    report.write(
                        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    )

                    for disease, history in st.session_state[
                        "severity_history"
                    ].items():

                        report.write(
                            f"{disease} -> {history}\n"
                        )

                st.success("Report Saved Successfully")


# ==========================================================
# VIDEO TAB
# ==========================================================
with video_tab:

    st.header("🎥 Upload Video")

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:

        if st.button("Process Video"):

            temp_video = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            temp_video.write(uploaded_video.read())

            cap = cv2.VideoCapture(temp_video.name)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            output_path = OUTPUT_VIDEO_DIR / "processed_video.mp4"

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (width, height)
            )

            progress_bar = st.progress(0)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

            current_frame = 0

            while True:

                success, frame = cap.read()

                if not success:
                    break

                result = detect_image(
                    model,
                    frame,
                    confidence_threshold
                )

                if len(result.boxes) > 0:

                    boxes = result.boxes.xyxy.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()

                    frame = draw_detections(
                        frame,
                        boxes,
                        classes,
                        confidences,
                        model.names,
                        recommendations,
                        show_overlay
                    )

                writer.write(frame)

                current_frame += 1

                progress_bar.progress(
                    min(
                        int(current_frame / total_frames * 100),
                        100
                    )
                )

            cap.release()
            writer.release()

            st.success("Video Processed Successfully")

            st.video(str(output_path))


# ==========================================================
# CHATBOT TAB
# ==========================================================

with chatbot_tab:

    st.header("🤖 Farmer Assistance Chatbot")

    user_query = st.text_input(
        "Describe crop symptoms"
    )

    if st.button("Get Suggestion"):

        if user_query.strip():
            response = chatbot_response(
                user_query,
                recommendations
            )
            st.success(response)
        else:
            st.warning("Please describe the symptoms first.")
