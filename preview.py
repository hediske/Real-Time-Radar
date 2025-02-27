import cv2
import argparse
import numpy as np
from LiveCapturing import LiveCapture, get_stream_infos

def get_coordinates(event, x, y, flags, param):
    """
    Callback function for mouse events. Prints the coordinates of a click.
    
    This function is passed to cv2.setMouseCallback() to capture mouse events.
    
    Parameters
    ----------
    event : int
        Type of event (e.g. cv2.EVENT_LBUTTONDOWN).
    x : int
        X-coordinate of the event.
    y : int
        Y-coordinate of the event.
    flags : int
        Additional flags.
    param : object
        Additional parameter.
    """
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
        print(f"Clicked at: ({x}, {y})")

def get_preview_frame(url, type="local"):
    """
    Retrieves a single frame from the given video source.

    Args:
        url (str): URL or path of the video source.
        type (str, optional): Type of video source. Defaults to "local".

    Returns:
        np.ndarray: The preview frame.
    """
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
            frame, stream_status = video_stream.read()
            if not stream_status:
                print("Stream ended. Exiting loop.")
                break
            if frame is not None:
                live.stop()
                break
    return frame

def display_preview(url, type="local"):
    """
    Displays a single frame from the given video source in a window.

    The frame is resized to 640x360 and displayed in a window with the title
    "Frame". The window has a mouse callback set to print the coordinates of the
    mouse click.

    Args:
        url (str): URL or path of the video source.
        type (str, optional): Type of video source. Defaults to "local".
    """
    frame = get_preview_frame(url, type)
    if frame is None:
        print("Error: Could not retrieve frame.")
        return
    cv2.imshow("Frame", cv2.resize(frame, (640, 360)))
    cv2.setMouseCallback("Frame", get_coordinates)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_source_frame(source, frame):
    
    """
    Resizes the frame to 640x360 and draws circles at each point in the source,
    along with a polygon connecting the points.

    Args:
        source (list of tuples): List of (x, y) coordinates for the source points.
        frame (np.ndarray): The input video frame.

    Returns:
        np.ndarray: The modified frame with drawn circles and polygon.
    """

    frame = cv2.resize(frame, (640, 360))
    for point in source:
        cv2.circle(frame, tuple(point), 10, (0, 0, 255), -1)
    frame = cv2.polylines(
        frame, [np.array(source)], isClosed=True, color=(255, 0, 0), thickness=2
    )
    return frame

def test_source(source, frame):
    """
    Displays a frame with the source points drawn as circles and a connecting polygon.

    Args:
        source (list of tuples): List of (x, y) coordinates for the source points.
        frame (np.ndarray): The input video frame.
    """
    frame = get_source_frame(source, frame)
    cv2.imshow("Frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    """
    CLI for Video Frame Preview

    This script provides a command-line interface for video frame processing. The
    available modes are 'preview' and 'test_source'. The 'preview' mode displays a
    single frame from the given video source (local or live). The 'test_source'
    mode displays a frame with the source points drawn as circles and a connecting
    polygon.

    Parameters:
        mode (str): Choose operation mode (preview or test_source)
        url (str): Video file path or stream URL
        type (str): Type of video source (local or live)
        source (list of int): List of points (x1 y1 x2 y2 ...) for source polygon

    Examples:
        python preview.py preview --url ./data/vehicles.mp4
        python preview.py test_source --url ./data/vehicles.mp4 --source 384 0 700 0 617 360 -50 360
    """
    parser = argparse.ArgumentParser(description="CLI for Video Frame Processing")
    parser.add_argument("mode", choices=["preview", "test_source"], help="Choose operation mode")
    parser.add_argument("--url", type=str, required=True, help="Video file path or stream URL")
    parser.add_argument("--type", type=str, choices=["local", "live"], default="local", help="Type of video source")
    parser.add_argument("--source", nargs='+', type=int, help="List of points (x1 y1 x2 y2 ...) for source polygon")
    
    args = parser.parse_args()
    
    if args.mode == "preview":
        display_preview(args.url, args.type)
    elif args.mode == "test_source":
        if args.source is None or len(args.source) % 2 != 0:
            print("Error: Source points must be provided as x1 y1 x2 y2 ...")
            return
        source_points = np.array(args.source).reshape(-1, 2)
        frame = get_preview_frame(args.url, args.type)
        if frame is not None:
            test_source(source_points, frame)
        else:
            print("Error: Could not retrieve frame.")

if __name__ == "__main__":
    main()
