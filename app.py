import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from VideoProcess import VideoProcessor
from preview import get_preview_frame, get_source_frame
from PIL import Image

# Initialize session state variables
if "processor" not in st.session_state:
    st.session_state.processor = None
if "current_model" not in st.session_state:
    st.session_state.current_model = None
if "current_iou" not in st.session_state:
    st.session_state.current_iou = None
if "current_confidence" not in st.session_state:
    st.session_state.current_confidence = None
if "processing" not in st.session_state:
    st.session_state.processing = False
if "temp_video_path" not in st.session_state:
    st.session_state.temp_video_path = None
if "current_frame" not in st.session_state:
    st.session_state.current_frame = None
if "source_points" not in st.session_state:
    st.session_state.source_points = []
if "target_points" not in st.session_state:
    st.session_state.target_points = []
def get_image_from_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)

def get_processor(model_type, confidence, iou):
    if (st.session_state.processor is None or 
        model_type != st.session_state.current_model):
        
        st.session_state.processor = VideoProcessor(model_type, confidence, iou)
    else:
        st.session_state.processor.setup_confidence(confidence)
        st.session_state.processor.setup_iou_threshod(iou)
    
    st.session_state.current_model = model_type
    st.session_state.current_confidence = confidence
    st.session_state.current_iou = iou

# Streamlit UI
st.title("Live Video Processing with YOLOv8")
st.sidebar.header("Adjust Parameters")

# Model Selection
model_type = st.sidebar.selectbox(
    "Select YOLOv8 Model",
    ["yolov8n-640", "yolov8n-1280", "yolov8l-640", "yolov8l-1280", "yolov8x-640", "yolov8x-1280"],
    index=4
)

# Sliders for confidence and IoU
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold", 0.1, 1.0, 0.7, 0.05)

options = {
    "YouTube Live (Live)": "live",
    "Local File (Local)": "local"
}

button_disabled = st.session_state.processing

if st.sidebar.button("Start Processing", disabled=button_disabled):
    st.session_state.processing = True
    try:
        with st.spinner('Loading the video Processor ...'):
            get_processor(model_type, confidence, iou_threshold)
    except Exception as e:
        st.session_state.processing = False
        st.error(f"Error loading the Video Processor: {e}")
    st.session_state.processing = False

if st.session_state.processor is None:
    st.error("Video Processor Not working. Please create a Processor")
else:
    st.markdown(
        f"""
        <div style="font-size: small;">
            <b>Current Model:</b> {st.session_state.current_model} | 
            <b>Confidence:</b> {st.session_state.current_confidence} | 
            <b>IoU Threshold:</b> {st.session_state.current_iou}
        </div>
        <br />
        <br /> 
        <br />

        """,
        unsafe_allow_html=True
    )

    # Video Souce Loading

    st.write("##### Video Souce Input")

    source_option = st.radio("Select Video Source", list(options.keys()))
    source_value = options[source_option]

    if source_value == "live":
        source = st.text_input("Enter YouTube URL")
    else:
        source = None
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            if st.session_state.temp_video_path is not None:
                os.remove(st.session_state.temp_video_path)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(uploaded_file.read())
                st.session_state.temp_video_path = temp_file.name
            source = st.session_state.temp_video_path
        else:
            source = None
    st.markdown(
        f"""
        <br />
        <br /> 
        <br />

        """,
        unsafe_allow_html=True
    )


    # Dynamic Polygon Coordinate Input
    st.write("##### Polygon Coordinate Input")
    col1, col2 = st.columns(2)
    # SOURCE Column
    with col1:
        st.write("###### SOURCE Coordinates")
        
        # Input for adding new SOURCE coordinate
        source_col1, source_col2 = st.columns(2)
        source_x_val = source_col1.number_input("X Coordinate", value=0, step=1, key="source_x_input")
        source_y_val = source_col2.number_input("Y Coordinate", value=0, step=1, key="source_y_input")

        if source_col1.button("Add SOURCE Coordinate"):
            st.session_state.source_points.append([source_x_val, source_y_val])
        # Button to clear SOURCE coordinates
        if source_col2.button("Clear SOURCE Coordinates"):
            st.session_state.source_points.clear()
        # Display SOURCE coordinates
        st.write("Current SOURCE Coordinates:")
        st.write(np.array(st.session_state.source_points))



    # TARGET Column
    with col2:
        st.write("###### TARGET Coordinates")
        
        # Input for adding new TARGET coordinate
        target_col1, target_col2 = st.columns(2)
        target_x_val = target_col1.number_input("X Coordinate", value=0, step=1, key="target_x_input")
        target_y_val = target_col2.number_input("Y Coordinate", value=0, step=1, key="target_y_input")

        if target_col1.button("Add TARGET Coordinate"):
            st.session_state.target_points.append([target_x_val, target_y_val])
        # Button to clear TARGET coordinates
        if target_col2.button("Clear TARGET Coordinates" , key="danger_button", help="This action is irreversible!"):
            st.session_state.target_points.clear()
        # Display TARGET coordinates
        st.write("Current TARGET Coordinates:")
        st.write(np.array(st.session_state.target_points))


    if st.button("Preview Video"):
        if source is None or source == "":
            st.error("Please provide a valid video source.")
        else:
            frame = get_preview_frame(source, source_value)
            st.session_state.current_frame = frame
            source_frame = get_source_frame(frame=st.session_state.current_frame, source=np.array(st.session_state.source_points))
            image_polygon = get_image_from_frame(source_frame)
            st.image(image_polygon, caption="Processed Image with Polygon Zone")

    if st.button("Process Video"):
        pass