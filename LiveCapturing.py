
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
        print(info)
        return {"url":info["url"], "fps":info["fps"], "width":info["width"], "height":info["height"]}
    

class LiveCapture:

    def __init__(self, url, max_buffer_size=100 , max_failures=50):
        self.url = url
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.stopped = False
        self.fail_count = 0
        self.stream_end = False
        self.max_failures = max_failures 
        if not self.isOpened():
            raise FileNotFoundError("Stream not found")
        self.frame_queue = queue.Queue(maxsize=max_buffer_size)

    def start(self):
        print('Started Streaming frames from the video stream')
        thread = threading.Thread(target=self.update, daemon=True)
        thread.start()
        return self

    def add_frame(self,frame):
        while True:
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame)  # Add frame to queue
                        break
                    else:
                        time.sleep(1)
    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            
            if ret:
                self.fail_count = 0 
                self.add_frame(frame)
            else:
                self.fail_count += 1
                print(f"Frame read failed. Failure count: {self.fail_count}/{self.max_failures}")
                if self.fail_count >= self.max_failures:
                    print("Max failures reached. Stopping capture.")
                    self.stream_end = True
                    self.stop()
                    break
                time.sleep(0.1)

    def stop(self):
        self.stopped = True
        self.stream_end = True
        self.cap.release()

    def read(self):
        try:
            return self.frame_queue.get(timeout=1),True
        except queue.Empty:
            if self.stream_end:
                return None,True
            return None,False
    
    def isOpened(self):
        print("Checking if the stream is opened")
        return self.cap.isOpened()    