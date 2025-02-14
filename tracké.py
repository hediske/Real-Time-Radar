import cv2
import yt_dlp

def get_live_stream_url(youtube_url):
    """Extracts the direct video URL of a YouTube live stream."""
    ydl_opts = {
        "format": "best",  # Get the best quality stream
        "quiet": True,  # Suppress console output
        "downloader_args": {"ffmpeg_i": "-http_persistent 0"},  # Add ffmpeg argument
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info["url"]  # Extract direct video URL

def stream_live_video(youtube_url):
    """Streams live video frames from YouTube and displays them using OpenCV."""
    stream_url = get_live_stream_url(youtube_url)
    
    cap = cv2.VideoCapture(stream_url)  # Open the video stream

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        cv2.imshow("Live Stream", frame)  # Show frame

        if cv2.waitKey(1) & 0xFF == ord("q"):  # Quit on 'q' key
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
live_url ="https://www.youtube.com/watch?v=5_XSYlAfJZM"
stream_live_video(live_url)
