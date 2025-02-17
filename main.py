
import os
import cv2
import yt_dlp
import threading
import queue
import time
from inference.models.utils import get_roboflow_model
import supervision as sv
from supervision.assets import VideoAssets, download_assets

# if not os.path.exists("data"):
#     os.makedirs("data")
# os.chdir("data")
# download_assets(VideoAssets.VEHICLES)

def get_stream_url(youtube_url):
    ydl_opts = {
        "format": "best",
        # "quiet": True,
        "noplaylist": True,
        "buffer_size": "16M",
        "downloader_args": {"ffmpeg_i": "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5"},
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        print(info)
        return info["url"]


class LiveCapture:

    def __init__(self, url,fps, max_buffer_size=100):
        self.url = url
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.stopped = False
        if not self.isOpened():
            raise FileNotFoundError("Stream not found")
        self.fps = fps
        self.frame_queue = queue.Queue(maxsize=max_buffer_size)

    def start(self):
        print('Started Streaming frames from the video stream')
        thread = threading.Thread(target=self.update, daemon=True)
        thread.start()
        return self

    def update(self):
        # fps = self.cap.get(cv2.CAP_PROP_FPS)
        # print(fps)
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)  # Add frame to queue
                else:
                    self.frame_queue.get()  # Remove oldest frame
                    self.frame_queue.put(frame)  # Add new frame

    def stop(self):
        self.stopped = True
        self.cap.release()

    def read(self):
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        return None  # Return None if queue is empty

    def isOpened(self):
        print("Checking if the stream is opened")
        return self.cap.isOpened()


class VideoTracker:
    def __init__(self, model_path):
        self.byte_track = None
        self.box_annotator = None
        self.label_annotator = None
        self.trace_annotator = None
        self.model = None
        self.load_model(model_path)

    def load_model(self, model_path):
        self.model = get_roboflow_model(model_path)
        print("Model Successfully Loaded")

    def setup_annotators(self, thickness, text_scale, fps):
        self.box_annotator = sv.BoxAnnotator(thickness=thickness)
        self.label_annotator = sv.LabelAnnotator(
            text_scale=text_scale,
            text_thickness=thickness,
            text_position=sv.Position.BOTTOM_CENTER,
        )
        self.trace_annotator = sv.TraceAnnotator(
            thickness=thickness,
            trace_length=fps * 2,
            position=sv.Position.BOTTOM_CENTER,
        )

    def setup_byte_tracker(self, fps):
        self.byte_track = sv.ByteTrack(
            minimum_matching_threshold=0.0,
            frame_rate=fps
        )

    def annotate_frame(self, frame):
        # Run model inference
        result = self.model.infer(frame)[0]
        detections = sv.Detections.from_inference(result)
        print(detections)

        # Use ByteTrack for tracking
        detections = self.byte_track.update_with_detections(detections=detections)
        print(detections)
        labels = [f"ID: {tracker_id}" for tracker_id in detections.tracker_id]

        # Annotate frame
        annotated_frame = self.box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = self.label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        return annotated_frame

    def stream_local_video(self, path):
        video_info = sv.VideoInfo.from_video_path(video_path=path)
        thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
        fps = video_info.fps

        self.setup_annotators(thickness, text_scale, fps)
        self.setup_byte_tracker(fps)

        frame_generator = sv.get_video_frames_generator(source_path=path)
        for frame in frame_generator:
            start_time = time.time()
            annotated_frame = self.annotate_frame(frame)
            annotated_frame = cv2.resize(annotated_frame, (640, 360))
            cv2.imshow('frame',annotated_frame)
            end_time = time.time()
            time.sleep(max(0, 1 / fps - (end_time - start_time)))
            if cv2.waitKey(1) & 0xFF == ord("q"):  # Quit on 'q' key
                break

model_path = "yolov8x-640"
video_path = "./data/vehicles.mp4"

tracker = VideoTracker(model_path)
tracker.stream_local_video(video_path)
