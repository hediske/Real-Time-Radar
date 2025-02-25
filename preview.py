import cv2
from LiveCapturing import LiveCapture, get_stream_infos
import supervision as sv
import numpy as np

def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
        print(f"Clicked at: ({x}, {y})")


def get_preview_frame(url, type="local") :
    frame = None
    if type == "local":
        cap = cv2.VideoCapture(url)
        ret = None
        while ret is None:
            ret, frame = cap.read()
        cap.release() 
    else:
        infos = get_stream_infos(url)
        live = LiveCapture(infos["url"])
        video_stream = live.start()
        while True:
            frame,stream_status = video_stream.read()
            if stream_status == False:
                print("Stream ended. Exiting loop.")
                break
            if frame is not None:
                live.stop()
                break 
    return frame    
    
def display_preview(url, type="local"):
    frame = get_preview_frame(url , type)
    if frame is None : 
        return
    cv2.imshow("Frame", cv2.resize(frame, (640, 360)))
    cv2.setMouseCallback("Frame", get_coordinates)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_source (source,frame):
    frame = get_source_frame(source , frame)
    cv2.imshow("Frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_source_frame(source,frame):
    frame = cv2.resize(frame, (640, 360))
    for point in source:
        cv2.circle(frame, tuple(point), 10, (0, 0, 255), -1)
    frame = cv2.polylines(
        frame, [source], isClosed=True, color=(255, 0, 0), thickness=2
    )
    return frame



# SOURCE = np.array([[229, 115], [351, 115], [920, 370] ,[-150, 370]])

# frame = get_preview_frame("./data/vehicles.mp4","local")
# # display_preview(frame)
# test_source(SOURCE,frame)