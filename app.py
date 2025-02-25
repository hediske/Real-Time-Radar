import time
from streamlit_image_coordinates import streamlit_image_coordinates
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from VideoProcess import VideoProcessor
from preview import get_preview_frame, get_source_frame
from PIL import Image
import queue
import threading
# Initialize session state variables
if "processor" not in st.session_state:
    st.session_state.processor = None
if "current_model" not in st.session_state:
    st.session_state.current_model = None
if "current_iou" not in st.session_state:
    st.session_state.current_iou = None
if "current_confidence" not in st.session_state:
    st.session_state.current_confidence = None
if "processing_button" not in st.session_state:
    st.session_state.processing_button = False
if "temp_video_path" not in st.session_state:
    st.session_state.temp_video_path = None
if "current_frame" not in st.session_state:
    st.session_state.current_frame = None
if "source_points" not in st.session_state:
    st.session_state.source_points = []
if "target_points" not in st.session_state:
    st.session_state.target_points = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "stop_processing" not in st.session_state:
    st.session_state.stop_processing = False
if "processed_video_path" not in st.session_state:
    st.session_state.processed_video_path = None
def get_image_from_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)

def callback():
    st.session_state.processor.stop_processor()
    st.session_state.stop_processing = True
    st.error("Stopped Streaming")

def callback_hide(show_hide):
    return not show_hide 
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

button_disabled = st.session_state.processing_button

if st.sidebar.button("Start Processing", disabled=button_disabled):
    st.session_state.processing_button = True
    try:
        with st.spinner('Loading the video Processor ...'):
            get_processor(model_type, confidence, iou_threshold)
    except Exception as e:
        st.session_state.processing_button = False
        st.error(f"Error loading the Video Processor: {e}")
    st.session_state.processing_button = False

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


    if st.button("Preview Video", disabled=st.session_state.processing):
        if source is None or source == "":
            st.error("Please provide a valid video source.")
        else:
            frame = get_preview_frame(source, source_value)
            st.session_state.current_frame = frame
    if  st.session_state.current_frame is not None:      
        source_frame = get_source_frame(frame=st.session_state.current_frame, source=np.array(st.session_state.source_points))
        image_polygon = get_image_from_frame(source_frame)
        # st.image(image_polygon, caption="Processed Image with Polygon Zone")
        value = streamlit_image_coordinates(image_polygon,key="pil")
        print(value)
        if value is not None:
            point = [value["x"], value["y"]]
            if point not in st.session_state.source_points:
                st.session_state.source_points.append(point)
                st.rerun()
                st.session_state["pil"] = None

    if st.button("Process Video", disabled=st.session_state.processing):
        if source is None or source == "":
            st.error("Please provide a valid video source.")
        else:
            st.session_state.processor.setup_source(np.array(st.session_state.source_points))
            st.session_state.processor.setup_target(np.array(st.session_state.target_points))

            # Start video processing in a separate thread
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
                st.session_state.processed_video_path = temp_video_file.name

            processing_thread = threading.Thread(
                target=st.session_state.processor.stream_video,
                args=(source_value, source, st.session_state.processed_video_path),
            )
            processing_thread.start()
            infos = None
            st.session_state.stop_processing = False
            # Show a spinner while waiting for infos
            with st.spinner("Fetching video information..."):
                while True:
                    try:
                        st.session_state.processor.start_processor()
                        # Wait for infos to be populated (timeout after 10 seconds)
                        infos = st.session_state.processor.infos_queue.get(timeout=30)
                        if infos is None:
                            st.error("Failed to fetch video information.")
                            st.session_state.processor.stop_processor()
                        else:
                            st.success("Video information fetched successfully!")
                        st.write("Video Information:", infos)
                    except queue.Empty:
                        st.error("Timed out waiting for video information.")
                        st.session_state.processor.stop_processor()
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                    break
                
                # Display processed frames
                queue = st.session_state.processor.get_frame_generator()
                #Adding Frame Display in Real Time
                st.button("Stop Processor", on_click=callback)
                show_hide = True
                if st.button("Show / Hide", on_click=callback_hide(show_hide)):
                    pass
                is_live = infos["is_live"]
                total_frames = infos["total_frames"]

                if not is_live:
                    progress_bar = st.progress(0)
                else :
                    st.write("Live Video")

                image = st.image([])
                placeholder = st.empty()
                processed_frames = 0


                while not st.session_state.stop_processing:
                    if not queue.empty():
                        frame = queue.get()
                        image_frame = get_image_from_frame(frame)
                        if show_hide and image is not None:
                            placeholder.image(image_frame, caption="Processed Frame")
                        else:
                            placeholder.empty() 
                        if not is_live and total_frames > 0:
                            processed_frames += 1
                            res = min(processed_frames / total_frames, 1.0)
                            progress_bar.progress(res, f"{res * 100:.2f}%")
                if not is_live and st.session_state.processed_video_path:
                    st.download_button("Download Processed Video", open(st.session_state.processed_video_path, "rb"), file_name="processed_video.mp4")


