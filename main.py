from VideoProcess import VideoProcessor


if __name__ == "__main__":
    model_path = "yolov8x-640"
    video_processor = VideoProcessor(model_path)

    # YOUTUBE_URL = "https://youtu.be/wqctLW0Hb_0?list=PLcQZGj9lFR7y5WikozDSrdk6UCtAnM9mB"
    # video_processor.stream_live_video(YOUTUBE_URL)

    path = "./data/vehicles.mp4"
    video_processor.stream_local_video(path)

