from VideoProcess import VideoProcessor
import numpy as np
SOURCE = np.array([[229, 115], [351, 115], [920, 370] ,[-150, 370]])


if __name__ == "__main__":
    model_path = "yolov8x-640"
    video_processor = VideoProcessor(model_path,source = SOURCE)

    # YOUTUBE_URL = "https://youtu.be/wqctLW0Hb_0?list=PLcQZGj9lFR7y5WikozDSrdk6UCtAnM9mB"
    # YOUTUBE_URL = "https://youtu.be/Gr0HpDM8Ki8"
    # video_processor.stream_live_video(YOUTUBE_URL,target = "./data/output2.mp4")

    path = "./data/vehicles.mp4"
    video_processor.stream_local_video(path,"./data/output.mp4")

