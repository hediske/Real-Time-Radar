import streamlit as st
import cv2
import numpy as np
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


def get_image_from_frame (frame):
    # FRAME_WINDOW = st.image([])
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame)
    # FRAME_WINDOW.image(pil_img)
    return pil_img

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

# Disable button while processing
button_disabled = st.session_state.processing

# Button to start processing
if st.sidebar.button("Start Processing", disabled=button_disabled):
    st.session_state.processing = True
    try:
        with st.spinner('Loading the video Processor ...'):
            get_processor(model_type, confidence, iou_threshold)
    except Exception:
        st.session_state.processing = False
        st.error("Error loading the Video Processor! Try again later")
    st.session_state.processing = False

if st.session_state.processor is None:
    st.error("Video Processor Not working. Please create a Processor")
else:
    st.success("Video Processor is UP")
    st.write("Current Model Type:", st.session_state.current_model)
    st.write("Current Confidence for the ByteTrack Algorithm:", st.session_state.current_confidence)
    st.write("Current IoU Threshold", st.session_state.current_iou)

    # Input for video source (YouTube or local path)
    source_option = st.radio("Select Video Source", list(options.keys()))
    source_value = options[source_option]

    # If YouTube is selected, take a URL input
    if source_value == "live":
        source = st.text_input("Enter YouTube URL")
    else:
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        source = None

    if st.button("Preview Video"):
        frame = get_preview_frame(source, source_value)
        if st.button("Show Polygon Zone"):
            source_frame = get_source_frame(frame, np.array([[229, 115], [351, 115], [920, 370], [-150, 370]]))
            # source_frame = cv2.cvtColor(source_frame, cv2.COLOR_BGR2RGB)
            st.image(source_frame, caption="Processed Image")
        else:
            image_preview = get_image_from_frame(frame)
            PREVIEW = st.image(image_preview)

    
    if st.button("Process Video"):
        pass
