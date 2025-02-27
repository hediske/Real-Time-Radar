from collections import defaultdict, deque
import cv2
import time
from LiveCapturing import LiveCapture, get_stream_infos
from inference.models.utils import get_roboflow_model
import supervision as sv
from supervision.utils.video import VideoInfo
import numpy as np
from queue import Queue
from tqdm import tqdm
from ViewTransformer import ViewTransformer
    


class VideoProcessor:
    def __init__(self, model_path ="yolov8n-640", source = None , target = None , iou_threshold = 0.3 ,confidence = 0.3):
        self.iou = iou_threshold
        self.source = source
        self.target = target 
        self.confidence = confidence
        self.byte_track = None
        self.box_annotator = None
        self.label_annotator = None
        self.trace_annotator = None
        self.coordinates = None
        self.polygon = None
        self.infos = None
        self.stopped = False
        self.view_transformer = None
        self.infos_queue = Queue()
        self.model = self.setup_model(model_path)
        self.frame_queue = Queue(maxsize=100)
    
    def get_frame_generator(self):
        return self.frame_queue

    def setup_coordinates(self,fps):
        self.coordinates = defaultdict(lambda: deque(maxlen=fps))

    def setup_confidence(self,confidence):
        self.confidence = confidence

    def setup_iou_threshod(self,iou):
        self.iou = iou

    def stop_processor(self):
        self.stopped = True

    def start_processor(self):
        self.stopped = False

    def setup_source(self,source):
        self.source = source

    def setup_view_transfromer(self):
        if self.target is not None and self.source is not None and  len(self.source) > 0 and len(self.target) > 0:
            self.view_transformer = ViewTransformer(self.source, self.target)
    def setup_target(self,target):
        self.target= target

    def setup_model(self,model_path):
        print(f"Setting the Model {model_path}")
        model = get_roboflow_model(model_path)
        print("Model Loaded Succesfully")
        return model
    
    def setup_annotators(self, thickness, text_scale, fps):
        """
        Set up annotators for drawing boxes, labels, and traces on video frames.

        Parameters
        ----------
        thickness : int
            The thickness of the annotations drawn on the frames.
        text_scale : float
            The scale of the text used in the label annotations.
        fps : int
            The frame rate of the video, used to calculate trace length.

        Notes
        -----
        The BoxAnnotator draws bounding boxes around detected objects.
        The LabelAnnotator adds labels with text scale and thickness.
        The TraceAnnotator draws traces with a length based on the frame rate.
        """
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
        """
        Set up the ByteTrack object for tracking objects.

        Parameters
        ----------
        fps : int
            The frame rate of the video stream.

        Notes
        -----
        The track_activation_threshold is set to the confidence level of the model.
        The lost_track_buffer is set to 50, which means that the tracker will hold onto
        a track for 50 frames after the object has left the frame.
        """
        self.byte_track = sv.ByteTrack(
            track_activation_threshold=self.confidence,
            lost_track_buffer=50,
            frame_rate=fps
        )

    def get_labels(self,points,tracker_ids,fps):
        """
        Generate labels for tracked objects including their speed estimates.

        This method calculates the speed of tracked objects based on their vertical
        movement over a series of frames. If points are provided, it appends the
        vertical coordinate to the respective tracker's history. The speed is
        computed only if the number of collected frames is sufficient (at least
        one third of the given frames per second). The speed is estimated in km/h.

        Args:
            points (list of lists): A list of [x, y] coordinates for the detected
                objects' anchors. If None, only tracker IDs are used.
            tracker_ids (list): A list of unique tracker IDs for the detected objects.
            fps (int): The frames per second of the video stream.

        Returns:
            list: A list of labels containing tracker IDs and speed estimates in
            the format "#{tracker_id} {speed} km/h". If speed cannot be estimated,
            only the tracker ID is returned.
        """
        if points is None :
            return [f"#{tracker_id}" for tracker_id in tracker_ids]
        for tracker_id, [_, y] in zip(tracker_ids, points):
                self.coordinates[tracker_id].append(y)
        labels = []
        for tracker_id in tracker_ids:
            # Ignoring The frames unless one third of fps number is collected
            if len(self.coordinates[tracker_id]) < fps / 3:
                labels.append(f"#{tracker_id}")
            else:
                coordinate_start = self.coordinates[tracker_id][-1]
                coordinate_end = self.coordinates[tracker_id][0]
                distance = abs(coordinate_start - coordinate_end)
                time = len(self.coordinates[tracker_id]) / fps
                speed = distance / time * 3.6
                labels.append(f"#{tracker_id} {int(speed)} km/h")
        return labels

    def clear_queue(self):
        """
        Clear the frame queue.

        This method is used to clear the frame queue. This is useful when the user
        changes the source video or the model, and we want to clear the old frames
        from the queue.

        This method is thread-safe, as it acquires the queue's lock before clearing
        the queue.
        """
        with self.frame_queue.mutex:  # Acquire the queue's lock
            self.frame_queue.queue.clear()

    def setup_polygon(self):
        """
        Setup the polygon area given the source points. This method will be called 
        each time the source points are updated. The polygon area is used to filter 
        the detections and remove the ones that are not inside the polygon area.
        """
        if self.source is not None and len(self.source) > 0:
            print(f"Setting the Polygon Area with : {self.source}")
            self.polygon = sv.PolygonZone(self.source)

    def annotate_frame(self, frame , fps):
        # Run model inference
        _start = time.time()
        result = self.model.infer(frame)[0]
        #Getting the Detections and filtering them
        detections = sv.Detections.from_inference(result)
        detections = detections[detections.confidence > self.confidence]
        mask = np.isin(detections.class_id, [7, 2])
        detections = detections[mask]

        #Non max merging for overlaps
        detections = detections.with_nmm(self.iou)

        #Add polygon filtering
        if self.polygon is not None : 
            detections = detections[self.polygon.trigger(detections=detections)]

        # Use ByteTrack for tracking
        detections = self.byte_track.update_with_detections(detections=detections)


        # Getting the points for the detections and speed Estimation
        points = None
        if self.view_transformer is not None  :
            points = detections.get_anchors_coordinates(
                            anchor=sv.Position.BOTTOM_CENTER)
            points = self.view_transformer.transform_points(points=points).astype(int)
        #Getting the labels and speed claculation
        labels = self.get_labels(points, detections.tracker_id,fps)

        # Annotate frame
        annotated_frame = self.box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = sv.draw_polygon(scene=annotated_frame,polygon=self.source)
        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        annotated_frame = self.trace_annotator.annotate(scene=annotated_frame, detections=detections)
        return annotated_frame


    def process_frame(self, frame, fps , display = False):
        start_time = time.time()
        frame  =  cv2.resize(frame, (640, 360))
        annotated_frame = self.annotate_frame(frame,fps)

        if not self.frame_queue.full():
            self.frame_queue.put(annotated_frame)

        if display == True:
            cv2.imshow("Local Video",annotated_frame )
            end_time = time.time()
            time.sleep(max(0, 1 / fps - end_time + start_time))
        return annotated_frame

    def stream_live_video(self, youtube_url,target = None, display= False):
        self.clear_queue()
        try:
            infos = get_stream_infos(youtube_url)
            if infos is None:
                raise ValueError("Failed to fetch video information.")
            self.infos = infos
            print(infos)
            self.infos_queue.put(infos) 
            thickness = sv.calculate_optimal_line_thickness(
                resolution_wh=(infos["height"], infos["width"])
            )
            text_scale = sv.calculate_optimal_text_scale(resolution_wh=(infos["width"], infos["height"]))
            fps = infos["fps"]
            self.setup_annotators((int)(thickness/5), text_scale, fps)
            self.setup_polygon()
            self.setup_view_transfromer()
            self.setup_byte_track(fps)
            self.setup_coordinates(fps)
            video_stream = LiveCapture(infos["url"]).start()
            
            
            total_frames = infos["total_frames"]
            is_live = infos["is_live"]


            if target is not None:
                infos = VideoInfo(infos["width"],infos["height"],infos["fps"],infos["total_frames"])
                with sv.VideoSink(target_path=target, video_info=infos) as sink:
                    progress_bar = tqdm(total=total_frames, desc="Processing Video") if not is_live else None

                    while True:
                        if self.stopped == True :
                            print("Exiting streaming. Processor Stopped !")
                            break
                        frame,stream_status = video_stream.read()
                        if stream_status == False:
                            print("Stream ended. Exiting loop.")
                            break
                        if frame is not None:
                            annotated_frame= self.process_frame(frame,fps,display)
                            sink.write_frame(annotated_frame)
                            if not is_live:
                                progress_bar.update(1) 
                        if cv2.waitKey(1) & 0xFF == ord("q"):  # Quit on 'q' key
                                break
                        if not is_live and progress_bar.n >= total_frames:
                            print("Processing completed.")
                            break
                    if not is_live:
                        progress_bar.close()
                    cv2.destroyAllWindows()

            else:
                progress_bar = tqdm(total=total_frames, desc="Processing Video") if not is_live else None

                while True:
                    if self.stopped == True :
                        print("Exiting streaming. Processor Stopped !")
                        break                    
                    frame,stream_status = video_stream.read()
                    if stream_status == False:
                        print("Stream ended. Exiting loop.")
                        break
                    if frame is not None:
                        self.process_frame(frame,fps,display)
                    
                        if not is_live:
                            progress_bar.update(1)

                    if cv2.waitKey(1) & 0xFF == ord("q"):  # Quit on 'q' key
                        break
                    if not is_live and progress_bar.n >= total_frames:
                        print("Processing completed.")
                        break
                if not is_live:
                    progress_bar.close()
                cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error in stream_live_video: {e}")
            self.infos_queue.put(None)

    def stream_local_video(self, path,target = None,display = False):
        self.clear_queue()
        try:
            video_infos = sv.VideoInfo.from_video_path(video_path=path)
            video_infos = VideoInfo(640,360,video_infos.fps,video_infos.total_frames)
            self.infos = video_infos
            print(video_infos)
            self.infos_queue.put({"width":video_infos.width,"height":video_infos.height,"fps":video_infos.fps,"total_frames":video_infos.total_frames , "is_live": False}) 
            thickness = sv.calculate_optimal_line_thickness(
                resolution_wh=video_infos.resolution_wh
            )
            text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_infos.resolution_wh)
            fps = video_infos.fps
            self.setup_annotators((int)(thickness/5), text_scale, fps)
            self.setup_polygon()
            self.setup_view_transfromer()
            self.setup_byte_track(fps)
            self.setup_coordinates(fps)
            frame_generator = sv.get_video_frames_generator(source_path=path)

            if target is not None:
                with sv.VideoSink(target_path=target, video_info=video_infos) as sink:
                    for frame in tqdm(frame_generator,total=video_infos.total_frames):
                        if self.stopped == True :
                            print("Exiting streaming. Processor Stopped !")
                            break  
                        annotated_frame = self.process_frame(frame,fps,display)
                        sink.write_frame(annotated_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):  # Quit on 'q' key
                            break
                    cv2.destroyAllWindows()

            else:
                for frame in tqdm(frame_generator,total=video_infos.total_frames):
                    if self.stopped == True :
                            print("Exiting streaming. Processor Stopped !")
                            break  
                    self.process_frame(frame, fps,display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):  # Quit on 'q' key
                        break
                cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error in stream_live_video: {e}")
            self.infos_queue.put(None)

    def stream_video(self , type, path , target = None ,display = False):
        if(type == 'live'):
            self.stream_live_video(path , target, display)
        else : 
            self.stream_local_video(path , target ,display)