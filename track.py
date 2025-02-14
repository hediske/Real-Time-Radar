
import cv2
from videoCapturing import VideoCapture , get_stream_url
from inference import get_model
import supervision as sv

def getModel():
    print("Loading the Model")
    model = get_model(model_id="yolov8n-640")
    print("Model Successully Loaded")
    return model

def annotate_frame(frame,model):
    frame = cv2.resize(frame, (640, 360))
    result = model.infer(frame)[0]
    descriptions = sv.Detections.from_inference(result)
    annotated_frame = frame.copy()

    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_frame = bounding_box_annotator.annotate(scene=annotated_frame, detections=descriptions)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=descriptions)   
   
    return annotated_frame



def stream_live_video(youtube_url,model):
    url = get_stream_url(youtube_url)
    video_stream = VideoCapture(url).start()
    while True:
        frame = video_stream.read()
        if frame is not None:  
            annotated_frame = annotate_frame(frame,model)   
            cv2.imshow("Live Stream", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):  # Quit on 'q' key
            break

    video_stream.stop()
    cv2.destroyAllWindows()


YOUTUBE_URL = "https://www.youtube.com/watch?v=5_XSYlAfJZM"
model = getModel()
stream_live_video(YOUTUBE_URL , model)