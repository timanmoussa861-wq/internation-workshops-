import streamlit as st
import pandas as pd
from PIL import Image
from collections import Counter

# YOLO
from ultralytics import YOLO


# =====================================================
# CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="YOLO Object Detector",
    page_icon="🔎",
    layout="wide"
)


# =====================================================
# TITLE
# =====================================================

st.title("🔎 YOLO Object Detection & Counter")

st.write(
    "Upload an image to detect, count and analyze objects."
)


# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_yolo_model():

    model = YOLO("yolo11n.pt")

    return model


# =====================================================
# UPLOAD IMAGE
# =====================================================

uploaded_file = st.file_uploader(
    "📷 Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)


# =====================================================
# MAIN APPLICATION
# =====================================================

if uploaded_file is not None:

    # -------------------------------------------------
    # OPEN IMAGE
    # -------------------------------------------------

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # -------------------------------------------------
    # DISPLAY ORIGINAL IMAGE
    # -------------------------------------------------

    st.subheader("📷 Original Image")

    st.image(
        image,
        use_container_width=True
    )


    # -------------------------------------------------
    # LOAD YOLO
    # -------------------------------------------------

    with st.spinner("Loading YOLO model..."):

        model = load_yolo_model()


    # -------------------------------------------------
    # DETECTION
    # -------------------------------------------------

    with st.spinner("🤖 Detecting objects..."):

        results = model.predict(
            source=image,
            conf=0.25,
            verbose=False
        )


    result = results[0]


    # -------------------------------------------------
    # DRAW DETECTIONS
    # -------------------------------------------------

    detected_image = result.plot()

    # Convert BGR → RGB
    detected_image = detected_image[:, :, ::-1]


    st.subheader("🎯 Detection Result")

    st.image(
        detected_image,
        use_container_width=True
    )


    # =================================================
    # GET DETECTED OBJECTS
    # =================================================

    detected_objects = []

    class_names = result.names


    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(
                box.cls[0].item()
            )

            object_name = class_names[
                class_id
            ]

            detected_objects.append(
                object_name
            )


    # =================================================
    # COUNT OBJECTS
    # =================================================

    object_counts = Counter(
        detected_objects
    )


    total_objects = len(
        detected_objects
    )


    different_types = len(
        object_counts
    )


    # =================================================
    # SUMMARY
    # =================================================

    st.divider()

    st.header("📊 Detection Summary")


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

            most_common_object = (
                object_counts
                .most_common(1)[0][0]
            )

        else:

            most_common_object = "None"


        st.metric(
            "Most Common",
            most_common_object
        )


    # =================================================
    # OBJECT INFORMATION
    # =================================================

    if total_objects > 0:

        st.subheader(
            "📋 Objects by Type"
        )


        data = []


        for object_type, quantity in (
            object_counts.most_common()
        ):

            percentage = (
                quantity
                / total_objects
                * 100
            )


            data.append({

                "Object Type": object_type,

                "Quantity": quantity,

                "Percentage": round(
                    percentage,
                    2
                )

            })


        df = pd.DataFrame(data)


        # -------------------------------------------------
        # TABLE
        # -------------------------------------------------

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )


        # -------------------------------------------------
        # BAR CHART
        # -------------------------------------------------

        st.subheader(
            "📈 Object Distribution"
        )


        chart_data = df.set_index(
            "Object Type"
        )["Quantity"]


        st.bar_chart(
            chart_data
        )


        # =================================================
        # CALCULATIONS
        # =================================================

        st.subheader(
            "🧮 Calculations"
        )


        st.write(
            f"Total objects detected: "
            f"**{total_objects}**"
        )


        st.write(
            f"Different object types: "
            f"**{different_types}**"
        )


        st.write("")


        for object_type, quantity in (
            object_counts.most_common()
        ):

            percentage = (
                quantity
                / total_objects
                * 100
            )


            st.write(
                f"• **{object_type}**: "
                f"{quantity} object(s) "
                f"→ **{percentage:.2f}%**"
            )


    else:

        st.warning(
            "⚠️ No objects were detected."
        )


# =====================================================
# NO IMAGE
# =====================================================

else:

    st.info(
        "👆 Please upload an image to start."
    )
