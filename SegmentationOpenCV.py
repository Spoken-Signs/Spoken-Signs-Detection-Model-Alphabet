"""
Per-frame landmark extraction for ASL data creation, using MediaPipe's
Tasks API (HolisticLandmarker, mediapipe >= 0.10.31).

Coordinates only: no pixels or image crops are kept.

Usage:
    extractor = LandmarkExtractor("clip.mp4")
    for out in extractor:
        out.frame_idx      # 0, 1, 2, ...
        out.timestamp_ms   # frame time in the video, in milliseconds
        out.landmarks      # {"face": ..., "right_hand": ..., "left_hand": ...}

Each landmark array has shape (num_points, 3) with columns x, y, z.
x and y are normalized to the frame (0 to 1); z is relative depth.
A value is None when that part wasn't detected in the frame.
"""
import os
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Sequence

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
             "holistic_landmarker/float16/latest/holistic_landmarker.task")


def _indices(*connection_sets) -> list:
    """Collect every landmark index used by the given MediaPipe connection sets."""
    return sorted({i for conns in connection_sets for c in conns for i in (c.start, c.end)})


_FACE = vision.FaceLandmarksConnections

# Optional face subsets. Pass one as `face_indices` to keep only those points.
LIPS = _indices(_FACE.FACE_LANDMARKS_LIPS)                                   # 40 points
LIPS_AND_BROWS = _indices(_FACE.FACE_LANDMARKS_LIPS,
                          _FACE.FACE_LANDMARKS_LEFT_EYEBROW,
                          _FACE.FACE_LANDMARKS_RIGHT_EYEBROW)                 # 60 points


@dataclass
class FrameOutput:
    """Landmarks extracted from one video frame."""
    frame_idx: int
    timestamp_ms: int
    # "face"       -> (478, 3), or (len(face_indices), 3) if a subset is set
    # "right_hand" -> (21, 3)
    # "left_hand"  -> (21, 3)
    landmarks: Dict[str, Optional[np.ndarray]]


class LandmarkExtractor:
    """
    Iterates over a video file and yields a FrameOutput for every frame.

    A fresh HolisticLandmarker is created each time you iterate, so tracking
    state never leaks from one video (or one pass) into another.
    """

    def __init__(
        self,
        video_path: str,
        model_path: str = "holistic_landmarker.task",
        face_indices: Optional[Sequence[int]] = None,  # None = all 478 face points
        min_confidence: float = 0.5,
    ):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        self.video_path = video_path
        self.model_path = self._ensure_model(model_path)
        self.face_indices = None if face_indices is None else list(face_indices)
        self.min_confidence = min_confidence

    def __iter__(self) -> Iterator[FrameOutput]:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        # Timestamps are derived from frame index and fps (deterministic,
        # unlike CAP_PROP_POS_MSEC, which varies between OpenCV backends)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        try:
            with vision.HolisticLandmarker.create_from_options(self._options()) as landmarker:
                frame_idx, last_ts = 0, -1
                while True:
                    ret, frame = cap.read()
                    if not ret:  # end of video
                        break

                    # VIDEO mode needs strictly increasing timestamps
                    ts = max(int(round(frame_idx * 1000 / fps)), last_ts + 1)
                    last_ts = ts

                    yield self._process_frame(landmarker, frame, frame_idx, ts)
                    frame_idx += 1
        finally:
            # Runs even if the caller breaks out of the loop early
            cap.release()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_model(path: str) -> str:
        """Download the model bundle (the neural network weights) on first use."""
        if not os.path.exists(path):
            print(f"Downloading model to {path} ...")
            urllib.request.urlretrieve(MODEL_URL, path)
        return path

    def _options(self) -> vision.HolisticLandmarkerOptions:
        return vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=vision.RunningMode.VIDEO,  # tracking across frames
            min_face_detection_confidence=self.min_confidence,
            min_pose_detection_confidence=self.min_confidence,
            min_hand_landmarks_confidence=self.min_confidence,
        )

    def _process_frame(self, landmarker, frame, frame_idx, ts) -> FrameOutput:
        # MediaPipe expects RGB; OpenCV reads BGR
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        face = self._to_array(result.face_landmarks)
        if face is not None and self.face_indices is not None:
            face = face[self.face_indices]

        landmarks = {
            "face": face,
            "right_hand": self._to_array(result.right_hand_landmarks),
            "left_hand": self._to_array(result.left_hand_landmarks),
        }
        return FrameOutput(frame_idx, ts, landmarks)

    @staticmethod
    def _to_array(lms) -> Optional[np.ndarray]:
        """List of NormalizedLandmark -> (N, 3) float32 array, or None if empty."""
        if not lms:
            return None
        return np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)


if __name__ == "__main__":
    import sys
    # just change this part to a for loop for dataset creation
    if len(sys.argv) != 2:
        sys.exit("usage: python asl_landmarks.py path/to/video.mp4")

    counts = {"face": 0, "right_hand": 0, "left_hand": 0}
    n = 0
    # Save it to a file
    for out in LandmarkExtractor(sys.argv[1]):
        n += 1
        for k, arr in out.landmarks.items():
            counts[k] += arr is not None
        shapes = {k: (None if v is None else v.shape) for k, v in out.landmarks.items()}
        print(f"frame {out.frame_idx:5d}  t={out.timestamp_ms:7d} ms  {shapes}")

    # Detection rate per part: a quick check of how often MediaPipe drops out
    if n:
        print("\nDetection rate:", {k: f"{100 * c / n:.1f}%" for k, c in counts.items()})