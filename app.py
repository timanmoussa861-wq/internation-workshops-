import streamlit as st
from PIL import Image
from collections import Counter
from ultralytics import YOLO
import pandas as pd


st.set_page_config(
    page_title="YOLO Object Detector",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 YOLO Object Detection & Counter")

st.write(
    "Upload an image and YOLO will detect and count the objects."
)


@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")


model = load_model()


uploaded_file = st.file_uploader(
    "📷 Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("📷 Original Image")
    st.image(image, use_container_width=True)

    with st.spinner("🤖 Detecting objects..."):

        results = model.predict(
            source=image,
            conf=0.25,
            verbose=False
        )

    result = results[0]

    # Image with bounding boxes
    annotated_image = result.plot()

    # Convert BGR → RGB
    annotated_image = annotated_image[:, :, ::-1]

    st.subheader("🎯 Detected Objects")
    st.image(
        annotated_image,
        use_container_width=True
    )

    # Get detected objects
    detected_objects = []

    names = result.names

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0].item())

            object_name = names[class_id]

            detected_objects.append(object_name)

    # Count objects
    object_counts = Counter(detected_objects)

    total_objects = len(detected_objects)

    different_types = len(object_counts)

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
            "Most Common",
            most_common
        )

    if total_objects > 0:

        # Create table
        data = []

        for object_type, quantity in object_counts.most_common():

            percentage = (
                quantity / total_objects
            ) * 100

            data.append({
                "Object Type": object_type,
                "Quantity": quantity,
                "Percentage": round(percentage, 2)
            })

        df = pd.DataFrame(data)

        st.subheader("📋 Objects by Type")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        # Chart
        st.subheader("📈 Object Distribution")

        chart = df.set_index(
            "Object Type"
        )["Quantity"]

        st.bar_chart(chart)

        # Calculations
        st.subheader("🧮 Calculations")

        st.write(
            f"Total objects detected: **{total_objects}**"
        )

        st.write(
            f"Different object types: **{different_types}**"
        )

        for object_type, quantity in object_counts.most_common():

            percentage = (
                quantity / total_objects
            ) * 100

            st.write(
                f"**{object_type}** = "
                f"{quantity} object(s) "
                f"→ {percentage:.2f}%"
            )

    else:

        st.warning(
            "⚠️ No objects were detected."
        )

else:

    st.info(
        "👆 Upload an image to start."
    )
