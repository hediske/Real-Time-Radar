import argparse

from VideoProcess import VideoProcessor

def process_video(
    type: str,  # 'local' or 'live'
    input: str,  # Path to the video file or YouTube URL
    output: str = None,  # Output file path (optional)
    source: list = None,  # Source polygon coordinates (optional)
    target: list = None,  # Target polygon coordinates (optional)
    model: str = "yolov8n-640",  # Model type (optional)
    iou: float = 0.3,  # IoU threshold (optional)
    confidence: float = 0.3,  # Confidence threshold (optional)
    display: bool = False,  # Whether to display the video (optional)
):
    """
    Process a video (local or live) using the specified parameters.

    Args:
        type (str): Type of video, either 'local' or 'live'.
        input (str): Path to the video file (for local) or YouTube URL (for live).
        output (str, optional): Path to save the output video. Defaults to None.
        source (list, optional): Source polygon coordinates. Defaults to None.
        target (list, optional): Target polygon coordinates. Defaults to None.
        model (str, optional): Model type for inference. Defaults to "yolov8x-640".
        iou (float, optional): IoU threshold for detection. Defaults to 0.3.
        confidence (float, optional): Confidence threshold for detection. Defaults to 0.3.
        display (bool, optional): Whether to display the video during processing. Defaults to False.
    """
    try:
        # Initialize the VideoProcessor with the specified parameters
        processor = VideoProcessor(
            model_path=model,
            source=source,
            target=target,
            iou_threshold=iou,
            confidence=confidence,
        )

        # Process the video based on its type
        if type == "live":
            processor.stream_live_video(input, target=output, display=display)
        elif type == "local":
            processor.stream_local_video(input, target=output, display=display)
        else:
            raise ValueError("Invalid video type. Must be 'local' or 'live'.")

    except Exception as e:
        print(f"Error processing video: {e}")

def parse_arguments():
    """
    Parse command-line arguments for the video processing script.
    """
    parser = argparse.ArgumentParser(description="Process a video (local or live) with optional parameters.")
    
    # Required arguments
    parser.add_argument("type", type=str, choices=["local", "live"], help="Type of video: 'local' or 'live'.")
    parser.add_argument("input", type=str, help="Path to the video file (for local) or YouTube URL (for live).")

    # Optional arguments
    parser.add_argument("--output", type=str, default=None, help="Path to save the output video.")
    parser.add_argument("--source", type=str, default=None, help="Source polygon coordinates as a string (e.g., '[[x1,y1],[x2,y2],...]').")
    parser.add_argument("--target", type=str, default=None, help="Target polygon coordinates as a string (e.g., '[[x1,y1],[x2,y2],...]').")
    parser.add_argument("--model", type=str, default="yolov8n-640", help="Model type for inference. Defaults to 'yolov8n-640'.")
    parser.add_argument("--iou", type=float, default=0.3, help="IoU threshold for detection. Defaults to 0.3.")
    parser.add_argument("--confidence", type=float, default=0.3, help="Confidence threshold for detection. Defaults to 0.3.")
    parser.add_argument("--display", action="store_true", help="Display the video during processing.")

    args = parser.parse_args()

    # Convert source and target from string to list of coordinates
    source = eval(args.source) if args.source else None
    target = eval(args.target) if args.target else None

    return args, source, target

if __name__ == "__main__":
    # Parse command-line arguments
    args, source, target = parse_arguments()

    # Call the process_video function with the parsed arguments
    process_video(
        type=args.type,
        input=args.input,
        output=args.output,
        source=source,
        target=target,
        model=args.model,
        iou=args.iou,
        confidence=args.confidence,
        display=args.display,
    )