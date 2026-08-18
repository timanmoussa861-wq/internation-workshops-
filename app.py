import streamlit as st
from PIL import Image
from collections import Counter
from ultralytics import YOLO
import pandas as pd


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="YOLO Object Counter",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 YOLO Object Detection & Object Counter")

st.write(
    "Upload an image and the application will detect objects, "
    "count each object type, and calculate percentages."
)


# -----------------------------
# Load YOLO model
# -----------------------------
@st.cache_resource
def load_model():
    model = YOLO("yolo11n.pt")
    return model


model = load_model()


# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)


# -----------------------------
# Process image
# -----------------------------
if uploaded_file:

    # Open uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    # -----------------------------
    # Object detection
    # -----------------------------
    with st.spinner("Detecting objects..."):

        results = model.predict(
            image,
            conf=0.25,
            verbose=False
        )

    result = results[0]

    # -----------------------------
    # Draw bounding boxes
    # -----------------------------
    annotated = result.plot()

    # Convert BGR to RGB
    annotated_image = Image.fromarray(
        annotated[..., ::-1]
    )

    st.subheader("Detected Objects")
    st.image(
        annotated_image,
        use_container_width=True
    )


    # -----------------------------
    # Get detected object names
    # -----------------------------
    names = result.names

    detected_objects = []

    if result.boxes is not None:

        for cls in result.boxes.cls.tolist():

            class_id = int(cls)

            object_name = names[class_id]

            detected_objects.append(object_name)


    # -----------------------------
    # Count objects
    # -----------------------------
    object_counts = Counter(detected_objects)

    total_objects = len(detected_objects)

    different_types = len(object_counts)


    # -----------------------------
    # Statistics
    # -----------------------------
    st.divider()

    st.subheader("📊 Detection Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Objects",
            total_objects
        )

    with col2:
        st.metric(
            "Different Types",
            different_types
        )

    with col3:

        if object_counts:

            most_common = object_counts.most_common(1)[0][0]

        else:

            most_common = "None"

        st.metric(
            "Most Common Object",
            most_common
        )


    # -----------------------------
    # Results table
    # -----------------------------
    if object_counts:

        rows = []

        for object_type, quantity in object_counts.most_common():

            percentage = (
                quantity / total_objects
            ) * 100

            rows.append({
                "Object Type": object_type,
                "Quantity": quantity,
                "Percentage": round(
                    percentage,
                    2
                )
            })


        dataframe = pd.DataFrame(rows)


        st.subheader("📋 Objects by Type")

        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True
        )


        # -----------------------------
        # Bar chart
        # -----------------------------
        st.subheader("📈 Object Distribution")

        chart_data = dataframe.set_index(
            "Object Type"
        )["Quantity"]

        st.bar_chart(chart_data)


        # -----------------------------
        # Calculations
        # -----------------------------
        st.subheader("🧮 Calculations")

        st.write(
            f"**Total detected objects:** "
            f"{total_objects}"
        )

        st.write(
            f"**Different object types:** "
            f"{different_types}"
        )


        for object_type, quantity in object_counts.most_common():

            percentage = (
                quantity / total_objects
            ) * 100

            st.write(
                f"**{object_type}**: "
                f"{quantity} object(s) "
                f"({percentage:.2f}%)"
            )


    else:

        st.warning(
            "No objects were detected. "
            "Try another image."
        )