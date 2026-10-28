# Object Detection and Tracking

Real-time object detection and tracking with **YOLO**, **SORT** and **OpenCV**. It finds objects in a webcam stream or a video file, draws a box around each one, and gives every object an ID that stays the same from frame to frame.

## How it works

1. **Video input (OpenCV)**: `cv2.VideoCapture` reads frames from your webcam or a video file.
2. **Detection (YOLO)**: a pre-trained YOLO11 model from the `ultralytics` package finds the objects in each frame. It recognizes 80 everyday classes such as person, car, bus, dog, bottle and laptop.
3. **Bounding boxes**: every detection gives a box, a class and a confidence score for each frame.
4. **Tracking (SORT)**: `sort.py` is a from-scratch implementation of SORT (Simple Online and Realtime Tracking). Each object gets a Kalman filter that predicts where it will be next. New detections are matched to those predictions by box overlap (IoU) using the Hungarian algorithm. A matched object keeps its ID, a new object gets a new ID, and an object missing for 20 frames is dropped.
5. **Display**: OpenCV shows the video live, with a colored box and a label such as `person #3` on every tracked object.

## Project structure

```
object-tracking/
├── track.py            main program: video input, detection, drawing, live display
├── sort.py             the SORT tracker
└── requirements.txt
```

## Setup (Windows)

You need **Python 3.10 to 3.13**.

```powershell
cd object-tracking
python -m pip install -r requirements.txt
```
This installs PyTorch, Ultralytics (YOLO) and OpenCV. It is a large download, so it takes a few minutes.

## Run

**Webcam**
```powershell
python track.py
```

**Video file**
```powershell
python track.py --source video.mp4
```

Press **Q** or **Esc** in the video window to quit. The first run downloads the YOLO model (about 5 MB), so it needs internet once.

Options:
- `--source 1` uses a second webcam, if you have one
- `--conf 0.5` shows only detections the model is at least 50% sure about (default 0.4)
- `--model yolo11s.pt` uses a larger, more accurate model that runs slower (default `yolo11n.pt`, the fastest)

## Speed

Everything runs on the CPU, so the speed depends on your computer. On a small 2-core test machine it ran at about 8 frames per second with the default model. A newer laptop is usually faster. If the video lags, close other programs or use a smaller video.

## Troubleshooting

- **A DLL error such as "DLL load failed" or "initialization routine failed"**: install the Microsoft Visual C++ runtime, close VS Code completely, reopen it and try again:
  ```powershell
  winget install Microsoft.VCRedist.2015+.x64
  ```
- **"Could not open webcam 0"**: close other apps that use the camera (Zoom, Teams, Camera), check **Settings → Privacy & security → Camera**, or try `--source 1`.
- **Objects get a new ID after being hidden for a while**: SORT only uses box position and motion, not how objects look. After 20 hidden frames an object is forgotten. Deep SORT adds appearance matching to handle this, at the cost of more complexity.
- **The same object flickers or is missed**: lower `--conf`, or try `--model yolo11s.pt`.

## Notes

- The tracker settings are at the top of the `Sort` class in `sort.py`: `max_age` (frames before an ID is dropped), `min_hits` (detections before a new object is shown) and `iou_threshold` (box overlap needed to count as the same object).
- Tracking is done per class, so a person is never matched with a car.
- Ultralytics is open source under the AGPL-3.0 license, which is fine for coursework and open-source projects. Check its terms before using it in a commercial product.
