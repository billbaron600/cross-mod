import cv2
from pathlib import Path

def save_video_frames(video_path: str):
    """
    Extract every frame from an .mp4 and save as PNGs.

    Args:
        video_path (str): Path to the input .mp4 file.

    Result:
        Creates a subfolder named after the video file
        (e.g. /path/to/my_clip/), and writes frame_0.png,
        frame_1.png, … inside it.
    """
    video_path = Path(video_path)
    out_dir = video_path.parent / video_path.stem      # e.g. /path/to/my_video/
    out_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(out_dir / f"frame_{idx}.png"), frame)
        idx += 1

    cap.release()
    print(f"Saved {idx} frames to {out_dir}")