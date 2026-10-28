"""Real-time object detection and tracking.

YOLO finds the objects in every video frame, SORT gives each object an ID that stays the same
from frame to frame, and OpenCV shows the result with boxes, labels and IDs.

Run:  python track.py                       (webcam)
      python track.py --source video.mp4    (video file)
Press Q or Esc in the video window to quit.
"""

import argparse
import os
import sys

import cv2
import numpy as np

from sort import Sort

WINDOW = "Object Detection and Tracking (press Q to quit)"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def open_source(source: str) -> cv2.VideoCapture:
    """Open a webcam (a number such as 0) or a video file."""
    if source.isdigit():
        index = int(source)
        # DirectShow opens webcams faster and more reliably on Windows
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(index)
        name = f"webcam {index}"
    else:
        capture = cv2.VideoCapture(source)
        name = source

    if not capture.isOpened():
        sys.exit(
            f"Could not open {name}.\n"
            "For a webcam: close other apps that use the camera, check Windows camera privacy settings, "
            "or try --source 1.\nFor a video: use an .mp4 file, for example --source video.mp4"
        )
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep the live image up to date (ignored if unsupported)
    return capture


def load_detector(weights: str):
    """Load the pre-trained YOLO model (downloaded automatically the first time)."""
    try:
        from ultralytics import YOLO

        return YOLO(weights)
    except Exception as error:
        sys.exit(
            f"Could not load the YOLO model: {error}\n"
            "If this mentions a DLL, install the Visual C++ runtime, then restart VS Code:\n"
            "    winget install Microsoft.VCRedist.2015+.x64\n"
            "If it mentions a download, check your internet connection."
        )


def detect(model, frame: np.ndarray, confidence: float) -> np.ndarray:
    """Find the objects in one frame. Returns rows of [x1, y1, x2, y2, confidence, class]."""
    boxes = model.predict(frame, conf=confidence, verbose=False)[0].boxes
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 6))
    return np.hstack(
        [
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy()[:, None],
            boxes.cls.cpu().numpy()[:, None],
        ]
    )


def color_for(track_id: int) -> tuple[int, int, int]:
    """A bright colour that stays the same for the same ID."""
    hue = (track_id * 37) % 180
    b, g, r = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def draw_tracks(frame: np.ndarray, tracks: np.ndarray, names: dict) -> None:
    """Draw a box with a label such as "person #3" for every tracked object."""
    for x1, y1, x2, y2, track_id, cls, _ in tracks:
        x1, y1, x2, y2 = (int(round(v)) for v in (x1, y1, x2, y2))
        color = color_for(int(track_id))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{names[int(cls)]} #{int(track_id)}"
        (text_w, text_h), baseline = cv2.getTextSize(label, FONT, 0.6, 2)
        top = max(y1 - text_h - baseline - 4, 0)  # keep the label inside the picture
        cv2.rectangle(frame, (x1, top), (x1 + text_w + 6, top + text_h + baseline + 4), color, -1)
        cv2.putText(frame, label, (x1 + 3, top + text_h + 2), FONT, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time object detection and tracking.")
    parser.add_argument("--source", default="0", help="webcam number (default 0) or path to a video file")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO weights (default yolo11n.pt, the fastest)")
    parser.add_argument("--conf", type=float, default=0.4, help="minimum detection confidence (default 0.4)")
    args = parser.parse_args()

    if not args.source.isdigit() and not os.path.exists(args.source):
        sys.exit(f"Video file not found: {args.source}")

    print("Loading the model (the first run downloads it, which can take a moment)...")
    model = load_detector(args.model)
    capture = open_source(args.source)
    tracker = Sort()

    print("Running. Press Q or Esc in the video window to quit.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("The video ended or the camera stopped.")
                break

            tracks = tracker.update(detect(model, frame, args.conf))
            draw_tracks(frame, tracks, model.names)

            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            window_closed = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
            if key in (ord("q"), 27) or window_closed:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
