import cv2
import threading
import queue
import time
from LiveCapturing import LiveCapture, get_stream_url
from inference import get_model
import supervision as sv

# Global ByteTrack instance (persistent tracking)
byte_track = sv.ByteTrack(minimum_matching_threshold=0.5,lost_track_buffer = 30)

# Frame queue for buffering
frame_queue = queue.Queue(maxsize=10)

def getModel():
    print("Loading the Model")
    model = get_model(model_id="yolov8n-640")
    print("Model Successfully Loaded")
    return model

def frame_reader(video_stream):
    """
    Continuously reads frames from the video stream into a buffer.
    """
    while True:
        frame = video_stream.read()
        if frame is not None:
            if not frame_queue.full():
                frame_queue.put(frame)
        time.sleep(0.01)  # Avoids CPU overload

def annotate_frame(frame, model):
    """
    Runs inference and tracks objects with ByteTrack.
    """
    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # Run model inference
    result = model.infer(frame)[0]
    detections = sv.Detections.from_inference(result)

    # Filter detections (ignore low confidence objects)
    detections = detections[detections.confidence > 0.3]

    # Use ByteTrack for tracking
    detections = byte_track.update_with_detections(detections=detections)

    labels = [f"ID: {tracker_id}" for tracker_id in detections.tracker_id]

    # Annotate frame
    annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    return annotated_frame

def stream_live_video(youtube_url, model):
    url = get_stream_url(youtube_url)
    video_stream = LiveCapture(url).start()

    # Start frame reader in a separate thread
    reader_thread = threading.Thread(target=frame_reader, args=(video_stream,))
    reader_thread.daemon = True
    reader_thread.start()
    print("Loading Frames")
    while True:
        print(frame_queue.qsize())
        if not frame_queue.empty():
            frame = frame_queue.get()
            frame = cv2.resize(frame, (640, 360))
            # annotated_frame = annotate_frame(frame, model)
            cv2.imshow("Live Stream", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):  # Quit on 'q' key
            break

    video_stream.stop()
    cv2.destroyAllWindows()

YOUTUBE_URL = "https://www.youtube.com/watch?v=5_XSYlAfJZM"
model = getModel()
stream_live_video(YOUTUBE_URL, model)
