import streamlit as st
import cv2
import numpy as np
from VideoProcess import VideoProcessor

# Streamlit UI
st.title("Live Video Processing with YOLOv8")
st.sidebar.header("Adjust Parameters")

# Sliders for confidence and IoU
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold", 0.1, 1.0, 0.3, 0.05)

# Input for video source (YouTube or local path)
source_option = st.sidebar.radio("Select Video Source", ("YouTube Live", "Local File"))
source = st.sidebar.text_input("Enter YouTube URL or File Path")

# Button to start processing
if st.sidebar.button("Start Processing"):
    st.sidebar.success("Processing Started...")
    try:
        processor = VideoProcessor(confidence=confidence, iou_threshold=iou_threshold)
    except Exception : 
        st.text("Error in the video opening path . Please verify the video link")
    if source_option == "YouTube Live":
        processor.stream_live_video(source)
    else:
        processor.stream_local_video(source)
