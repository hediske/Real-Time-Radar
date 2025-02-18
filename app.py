import streamlit as st
import cv2
import numpy as np
from VideoProcess import VideoProcessor
import tempfile


current_model = None
current_iou  = None 
current_confidence = None
processor : VideoProcessor | None = None


def get_processor(model_type, confidence , iou):
    if processor is None or not model_type == current_model: 
        return VideoProcessor(model_type,confidence, iou)
    else:
        processor.setup_confidence(confidence)  
        processor.setup_iou_threshod(iou)


# Streamlit UI
st.title("Live Video Processing with YOLOv8")
st.sidebar.header("Adjust Parameters")


# Model Selection
model_type = st.sidebar.selectbox(
    "Select YOLOv8 Model",
    ["yolov8n-640", "yolov8n-1280", "yolov8l-640", "yolov8l-1280", "yolov8x-640", "yolov8x-1280"],index=4
)   

# Sliders for confidence and IoU
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold", 0.1, 1.0, 0.3, 0.05)





# Initialize session state for button
if "processing" not in st.session_state:
    st.session_state.processing = False

# Disable button while processing
button_disabled = st.session_state.processing
# Button to start processing
if st.sidebar.button("Start Processing", disabled=button_disabled):
    st.session_state.processing = True
    try:
        with st.spinner('Loading the video Processor ...'):
            try:
                processor = get_processor(model_type,confidence,iou_threshold)
            except Exception :
                st.session_state.processing = False
                st.error("Error loading the Video Processor ! Try again later")
    except Exception :   
        st.error("Error loading the video , Please verify the vide path ")
        st.session_state.processing = False
    st.session_state.processing = False

if processor is  None :

    st.error("Video Processsor Not working . Please create a Processor")
else:
    st.success("Video Processsor is UP")
    st.write("Current Model Type :",model_type)    
    st.write("Current Confidence for the ByteTrack Algorithm :",confidence)
    st.write("Current Iou Threshold" , iou_threshold)
    # Input for video source (YouTube or local path)
    source_option = st.sidebar.radio("Select Video Source", ("YouTube Live", "Local File"))

    # If YouTube is selected, take a URL input
    if source_option == "YouTube Live":
        source = st.sidebar.text_input("Enter YouTube URL")

    # If Local File is selected, allow file upload
    else:
        uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        source = None
