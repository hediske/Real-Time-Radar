import cv2
from LiveCapturing import LiveCapture, get_stream_infos


def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
        print(f"Clicked at: ({x}, {y})")


def get_preview(url, type) :
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
    cv2.imshow("Frame", frame)
    cv2.setMouseCallback("Frame", get_coordinates)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


get_preview("./data/vehicles2.mp4","local")