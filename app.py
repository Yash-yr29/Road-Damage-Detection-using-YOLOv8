import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import cv2


st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🛣️",
    layout="wide"
)

st.title("🛣️ Road Damage Detection using YOLOv8")
st.write("Upload a road image to detect potholes and cracks.")

@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

model = load_model()

st.sidebar.header("Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.1,
    1.0,
    0.25,
    0.05
)

iou = st.sidebar.slider(
    "IoU Threshold",
    0.1,
    1.0,
    0.45,
    0.05
)

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    if st.button("🚀 Detect Road Damage"):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)

            results = model.predict(
                source=tmp.name,
                conf=confidence,
                iou=iou,
                save=False
            )

        result = results[0]

        plotted = result.plot()

        plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

        with col2:
            st.subheader("Detection Result")
            st.image(plotted, use_container_width=True)

        st.markdown("---")
        st.subheader("Detection Summary")

        boxes = result.boxes

        if len(boxes) == 0:

            st.success("✅ No road damage detected.")

        else:

            names = result.names

            for box in boxes:

                cls = int(box.cls)

                conf = float(box.conf)

                st.write(
                    f"**{names[cls]}** : {conf*100:.2f}%"
                )

        output_path = "outputs/result.jpg"

        os.makedirs("outputs", exist_ok=True)

        cv2.imwrite(
            output_path,
            cv2.cvtColor(plotted, cv2.COLOR_RGB2BGR)
        )

        with open(output_path, "rb") as file:

            st.download_button(
                label="📥 Download Result",
                data=file,
                file_name="road_damage_result.jpg",
                mime="image/jpeg"
            )