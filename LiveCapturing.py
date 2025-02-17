
import cv2
import yt_dlp
import threading
import queue
import time

def get_stream_infos(youtube_url):
    ydl_opts = {
        "format": "best",
        # "quiet": True,
        "noplaylist": True,
        "buffer_size": "16M",
        "downloader_args": {"ffmpeg_i": "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5"},
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return {"url":info["url"], "fps":info["fps"], "width":info["width"], "height":info["height"]}
    

class LiveCapture:

    def __init__(self, url, max_buffer_size=100):
        self.url = url
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.stopped = False
        if not self.isOpened():
            raise FileNotFoundError("Stream not found")
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
                while True:
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame)  # Add frame to queue
                        break
                    else:
                        time.sleep(10)

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