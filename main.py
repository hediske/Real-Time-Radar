from VideoProcess import VideoProcessor
import numpy as np
from preview import get_preview_frame,display_preview
SOURCE = np.array([[384, 0], [700, 0], [617, 360] ,[-50, 360]])
TARGET = np.array([[0,0], [12,0], [12,60], [0,60]])

if __name__ == "__main__":
    model_path = "yolov8n-640"
    video_processor = VideoProcessor(model_path,source = SOURCE , target = TARGET)

    # YOUTUBE_URL = "https://youtu.be/wqctLW0Hb_0?list=PLcQZGj9lFR7y5WikozDSrdk6UCtAnM9mB"
    YOUTUBE_URL = "https://youtu.be/Gr0HpDM8Ki8"
    video_processor.stream_live_video(YOUTUBE_URL,target = "./data/output2.mp4" , display = True)

    # path = "./data/vehicles2.mp4"
    # video_processor.stream_local_video(path,"./data/output.mp4")
    # display_preview(YOUTUBE_URL,"live")
