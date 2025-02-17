import cv2
import yt_dlp
import queue
import time
from LiveCapturing import LiveCapture, get_stream_infos
from inference.models.utils import get_roboflow_model
import supervision as sv



    


class VideoProcessor:
    def __init__(self, model_path ="yolov8x-640", target_width = 25, target_height = 250 , iou_threshold = 0.7 ,confidence = 0.3):
        self.iou = iou_threshold
        self.confidence = confidence
        self.target_width = target_width
        self.target_height = target_height
        self.byte_track = None
        self.box_annotator = None
        self.label_annotator = None
        self.trace_annotator = None
        self.model = self.setup_model(model_path)

    def setup_model(self,model_path):
        print(f"Setting the Model {model_path}")
        model = get_roboflow_model(model_path)
        print("Model Loaded Succesfully")
        return model

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

    def setup_byte_track(self, fps):
        self.byte_track = sv.ByteTrack(
            track_activation_threshold=self.confidence,
            lost_track_buffer=50,
            frame_rate=fps
        )

    def annotate_frame(self, frame):
        # Run model inference
        _start = time.time()
        result = self.model.infer(frame)[0]
        print("inference = ", time.time() - _start)
        detections = sv.Detections.from_inference(result)
        detections = detections[detections.confidence > self.confidence]
        #Non max merging for overlaps
        detections = detections.with_nmm(self.iou)
        # Use ByteTrack for tracking
        detections = self.byte_track.update_with_detections(detections=detections)

        labels = [f"ID: {tracker_id}" for tracker_id in detections.tracker_id]

        # Annotate frame
        annotated_frame = self.box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = self.label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        return annotated_frame


    def process_frame(self,frame,fps):
        start_time = time.time()
        annotated_frame = self.annotate_frame(frame)
        annotated_frame = cv2.resize(annotated_frame, (640, 360))
        cv2.imshow("Local Video", annotated_frame)
        end_time = time.time()
        time.sleep(max(0, 1 / fps - end_time + start_time))
        return annotated_frame

    def stream_live_video(self, youtube_url,target = None):
        infos = get_stream_infos(youtube_url)
        print(infos)
        thickness = sv.calculate_optimal_line_thickness(
            resolution_wh=(infos["height"], infos["width"])
        )
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=(infos["width"], infos["height"]))
        fps = infos["fps"]
        self.setup_annotators(thickness, text_scale, fps)
        self.setup_byte_track(fps)

        video_stream = LiveCapture(infos["url"]).start()

        if target is not None:
            with sv.VideoSink(target_path=target, video_info=infos) as sink:
                while True:
                    frame = video_stream.read()
                    if frame is not None:
                        annotated_frame= self.process_frame(frame,fps)
                        sink.write_frame(annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):  # Quit on 'q' key
                            break
        else:
            while True:
                frame = video_stream.read()
                if frame is not None:
                        self.process_frame(frame,fps)
                if cv2.waitKey(1) & 0xFF == ord("q"):  # Quit on 'q' key
                    break
        cv2.destroyAllWindows()

    def stream_local_video(self, path,target = None):
        video_infos = sv.VideoInfo.from_video_path(video_path=path)
        print(video_infos)
        thickness = sv.calculate_optimal_line_thickness(
            resolution_wh=video_infos.resolution_wh
        )
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_infos.resolution_wh)
        fps = video_infos.fps
        self.setup_annotators(thickness, text_scale, fps)
        self.setup_byte_track(fps)

        frame_generator = sv.get_video_frames_generator(source_path=path)

        if target is not None:
            with sv.VideoSink(target_path=target, video_info=video_infos) as sink:
                for frame in frame_generator:
                    annotated_frame = self.process_frame(frame,fps)
                    sink.write_frame(annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):  # Quit on 'q' key
                        break
        else:
            for frame in frame_generator:
                self.process_frame(frame, fps)
                if cv2.waitKey(1) & 0xFF == ord("q"):  # Quit on 'q' key
                    break
        cv2.destroyAllWindows()



