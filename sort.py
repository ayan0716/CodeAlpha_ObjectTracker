"""SORT: Simple Online and Realtime Tracking.

Every tracked object has its own Kalman filter, which predicts where its box will be in the
next frame. Each new frame, the detections are matched to those predictions by how much
their boxes overlap (IoU) using the Hungarian algorithm. A matched object keeps its ID.
An unmatched detection becomes a new object, and an object that is not seen for
`max_age` frames is dropped.

Based on: Bewley et al., "Simple Online and Realtime Tracking" (2016).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Overlap (intersection over union) between every box in A and every box in B."""
    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]
    width = np.maximum(0.0, np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]))
    height = np.maximum(0.0, np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]))
    intersection = width * height
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return intersection / np.maximum(area_a + area_b - intersection, 1e-9)


def box_to_state(box) -> np.ndarray:
    """[x1, y1, x2, y2] -> [center x, center y, area, width/height]"""
    w, h = box[2] - box[0], box[3] - box[1]
    return np.array([box[0] + w / 2, box[1] + h / 2, w * h, w / max(h, 1e-6)]).reshape(4, 1)


def state_to_box(state: np.ndarray) -> np.ndarray:
    """[center x, center y, area, width/height] -> [x1, y1, x2, y2]"""
    area, ratio = max(float(state[2, 0]), 1e-6), max(float(state[3, 0]), 1e-6)
    w = np.sqrt(area * ratio)
    h = area / w
    return np.array([state[0, 0] - w / 2, state[1, 0] - h / 2, state[0, 0] + w / 2, state[1, 0] + h / 2])


# Kalman filter settings. The state is [x, y, area, ratio, vx, vy, v_area]:
# the box position and size, plus how fast the position and area are changing.
_F = np.eye(7)  # next state = current state + velocity
_F[0, 4] = _F[1, 5] = _F[2, 6] = 1.0
_H = np.eye(4, 7)  # we only measure x, y, area and ratio, not the velocities
_R = np.eye(4)  # measurement noise
_R[2:, 2:] *= 10.0
_Q = np.eye(7)  # process noise
_Q[-1, -1] *= 0.01
_Q[4:, 4:] *= 0.01


class Track:
    """One tracked object."""

    def __init__(self, detection: np.ndarray, track_id: int):
        self.id = track_id
        self.cls = int(detection[5])
        self.conf = float(detection[4])
        self.x = np.zeros((7, 1))
        self.x[:4] = box_to_state(detection[:4])
        self.P = np.eye(7)  # uncertainty: very unsure about the initial velocities
        self.P[4:, 4:] *= 1000.0
        self.P *= 10.0
        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0

    def predict(self) -> np.ndarray:
        """Move the box forward one frame and return where we expect it to be."""
        if self.x[6, 0] + self.x[2, 0] <= 0:  # never let the area become negative
            self.x[6, 0] = 0.0
        self.x = _F @ self.x
        self.P = _F @ self.P @ _F.T + _Q
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return state_to_box(self.x)

    def update(self, detection: np.ndarray) -> None:
        """Correct the prediction with a matched detection."""
        self.cls = int(detection[5])
        self.conf = float(detection[4])
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        innovation = box_to_state(detection[:4]) - _H @ self.x
        S = _H @ self.P @ _H.T + _R
        gain = self.P @ _H.T @ np.linalg.inv(S)
        self.x = self.x + gain @ innovation
        self.P = (np.eye(7) - gain @ _H) @ self.P


class Sort:
    def __init__(self, max_age: int = 20, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        max_age        frames an object may go unseen before its ID is dropped
        min_hits       times an object must be detected before it is shown
        iou_threshold  minimum box overlap for a detection to count as the same object
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: list[Track] = []
        self.frame_count = 0
        self.next_id = 1

    def update(self, detections) -> np.ndarray:
        """Process one frame.

        detections: array of [x1, y1, x2, y2, confidence, class] rows (may be empty).
        Returns an array of [x1, y1, x2, y2, track id, class, confidence] rows.
        """
        detections = np.asarray(detections, dtype=float).reshape(-1, 6)
        self.frame_count += 1

        # 1. Predict where every known object is now
        predicted = np.array([track.predict() for track in self.tracks]).reshape(-1, 4)
        healthy = ~np.isnan(predicted).any(axis=1)
        self.tracks = [t for t, ok in zip(self.tracks, healthy) if ok]
        predicted = predicted[healthy]

        # 2. Match detections to predictions
        matches, unmatched = self._match(detections, predicted)
        for track_index, detection_index in matches:
            self.tracks[track_index].update(detections[detection_index])

        # 3. Every detection that matched nothing is a new object
        for detection_index in unmatched:
            self.tracks.append(Track(detections[detection_index], self.next_id))
            self.next_id += 1

        # 4. Report the objects seen in this frame. A new object is shown only after it has been
        #    matched `min_hits` times, which filters out one-frame false detections. Once shown,
        #    it stays visible whenever it is detected, so boxes do not flicker.
        results = []
        for track in self.tracks:
            confirmed = track.hits >= self.min_hits or self.frame_count <= self.min_hits
            if track.time_since_update < 1 and confirmed:
                results.append([*state_to_box(track.x), track.id, track.cls, track.conf])

        # 5. Forget objects that have been lost for too long
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return np.array(results, dtype=float).reshape(-1, 7)

    def _match(self, detections: np.ndarray, predicted: np.ndarray):
        """Pair detections with tracks by box overlap (only within the same object class)."""
        if len(self.tracks) == 0 or len(detections) == 0:
            return [], list(range(len(detections)))

        overlap = iou_matrix(detections[:, :4], predicted)
        track_classes = np.array([t.cls for t in self.tracks])
        overlap = overlap * (detections[:, 5][:, None] == track_classes[None, :])

        rows, cols = linear_sum_assignment(-overlap)  # Hungarian algorithm: best overall pairing
        matches, matched = [], set()
        for detection_index, track_index in zip(rows, cols):
            if overlap[detection_index, track_index] >= self.iou_threshold:
                matches.append((track_index, detection_index))
                matched.add(detection_index)
        return matches, [d for d in range(len(detections)) if d not in matched]
