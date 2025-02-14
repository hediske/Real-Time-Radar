
import cv2
import yt_dlp
import threading
import queue

def get_stream_url(youtube_url):
    ydl_opts = {
        "format": "best[ext=mp4]",
        "quiet": True,
        "noplaylist": True,
        "buffer_size": "16M",
        "downloader_args": {"ffmpeg_i": "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5"},
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info["url"]
    

class VideoCapture:

    def __init__(self, url, max_buffer_size=30):
        self.url = url
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.stopped = False
        self.frame_queue = queue.Queue(maxsize=max_buffer_size)

    def start(self):
        print('Started Streaming frames from the video stream')
        thread = threading.Thread(target=self.update, daemon=True)
        thread.start()
        return self

    def update(self):
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