import cv2
import threading
import queue
import time
from LiveCapturing import LiveCapture, get_stream_url
from inference import get_model
import supervision as sv

# Global ByteTrack instance (persistent tracking)
byte_track = None

# Frame queue for buffering
frame_queue = queue.Queue()

def getModel():
    print("Loading the Model")
    model = get_model(model_id="yolov8n-640")
    print("Model Successfully Loaded")
    return model


def annotate_frame(frame, model):
    """
    Runs inference and tracks objects with ByteTrack.
    """

    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # Run model inference
    _start = time.time()
    result = model.infer(frame)[0]
    print("inference = ",time.time()-_start)
    detections = sv.Detections.from_inference(result)

    # Use ByteTrack for tracking
    detections = byte_track.update_with_detections(detections=detections)

    labels = [f"ID: {tracker_id}" for tracker_id in detections.tracker_id]

    # Annotate frame
    # annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

    return annotated_frame

def stream_live_video(youtube_url, model):
    url = get_stream_url(youtube_url)
    global byte_track
    byte_track = sv.ByteTrack(minimum_matching_threshold=0.4,lost_track_buffer = 50)
    video_stream = LiveCapture(url).start()

    x=0
    while True:
        x = x +1
        if x%8 == 0:
            frame = video_stream.read()
            if frame is not None:

                frame = cv2.resize(frame, (640, 360))
                # annotated_frame = annotate_frame(frame, model)
                cv2.imshow("Live Stream", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):  # Quit on 'q' key
                break

    video_stream.stop()
    cv2.destroyAllWindows()

# YOUTUBE_URL = "https://youtu.be/wqctLW0Hb_0?list=PLcQZGj9lFR7y5WikozDSrdk6UCtAnM9mB"
model = getModel()
# stream_live_video(YOUTUBE_URL, model)


path = "./Road traffic video for object recognition.mp4"

video_infos = sv.VideoInfo.from_video_path(video_path=path)
print(video_infos)

byte_track = sv.ByteTrack(
    frame_rate=video_infos.fps)

thickness = sv.calculate_optimal_line_thickness(
    resolution_wh=video_infos.resolution_wh
)
text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_infos.resolution_wh)
box_annotator = sv.BoxAnnotator(thickness=thickness)
label_annotator = sv.LabelAnnotator(
  text_scale=text_scale,
  text_thickness=thickness,
  text_position=sv.Position.BOTTOM_CENTER,
)
trace_annotator = sv.TraceAnnotator(
  thickness=thickness,
  trace_length=video_infos.fps * 2,
  position=sv.Position.BOTTOM_CENTER,
)

frame_generator = sv.get_video_frames_generator(source_path=path)
for frame in frame_generator:
    start_time = time.time()
    annotated_frame = annotate_frame(frame, model)
    end_time = time.time()
    print(end_time-start_time)
    print("default : ",1/25)
    cv2.imshow("frame", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
                break
cv2.destroyAllWindows()

