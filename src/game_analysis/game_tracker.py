"""Multi-object tracker for game analysis.

Wraps ultralytics .track() with ByteTrack to maintain stable IDs
for all players, puck, goalies, and referees across frames.
Uses the HockeyAI model (7 classes).
"""

import numpy as np
from typing import Optional
from pathlib import Path

from .game_context import TrackedObject
from .puck_filter import PuckFilter


# HockeyAI model class mapping
HOCKEY_CLASSES = {
    0: "centroid",    # Center ice dot
    1: "faceoff",     # Faceoff dots
    2: "goal",        # Goal frame
    3: "goalie",      # Goaltender
    4: "player",      # Skater
    5: "puck",        # Puck
    6: "referee",     # Referee
}

# Which classes are "game objects" vs "rink landmarks"
GAME_OBJECT_CLASSES = {"player", "goalie", "puck", "referee"}
RINK_LANDMARK_CLASSES = {"centroid", "faceoff", "goal"}


class GameTracker:
    """Track all objects on the ice using HockeyAI YOLO + ByteTrack.

    Usage:
        tracker = GameTracker()
        for frame in video:
            objects = tracker.process_frame(frame, is_camera_cut=False)
            # objects is a list of TrackedObject with stable IDs
    """

    def __init__(
        self,
        model_path: str = "models/HockeyAI_model_weight.pt",
        conf: float = 0.25,
        iou: float = 0.5,
        tracker: str = "bytetrack.yaml",
    ):
        from ultralytics import YOLO

        model_file = Path(model_path)
        if not model_file.is_file():
            raise FileNotFoundError(
                f"HockeyAI model not found at {model_path}. "
                f"Download from HuggingFace: SimulaMet-HOST/HockeyAI"
            )

        self.model = YOLO(model_path)
        # Separate YOLO instance for the puck-only fallback. Calling predict()
        # on the same instance corrupts the ByteTrack state for subsequent
        # track() calls (track and predict share internal model state in
        # ultralytics). A second instance is the simplest robust fix.
        self.fallback_model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.tracker_config = tracker
        # Fallback puck-only confidence: lowered threshold for the fallback
        # YOLO call when the primary call returns no puck. The PuckFilter
        # still rejects bad candidates via on-ice + proximity scoring.
        self.puck_fallback_conf = 0.10

        # Verify class names match expected
        self.class_names = {}
        if hasattr(self.model, "names"):
            self.class_names = self.model.names
        print(f"  GameTracker: Model loaded with classes: {self.class_names}")

        # Cache the puck class id from the model's class names map for fast
        # filtered fallback predictions
        self._puck_class_id = None
        for cid, name in self.class_names.items():
            if "puck" in str(name).lower():
                self._puck_class_id = int(cid)
                break

        # Single-puck enforcement + on-ice filtering + short coast on dropouts
        self.puck_filter = PuckFilter()

        self._frame_count = 0

    def process_frame(
        self,
        frame: np.ndarray,
        is_camera_cut: bool = False,
    ) -> list:
        """Track all objects in one frame.

        Args:
            frame: BGR video frame.
            is_camera_cut: If True, reset tracker IDs (camera angle changed).

        Returns:
            List of TrackedObject with stable track_ids.
        """
        self._frame_count += 1

        # Run tracking (ByteTrack maintains IDs across frames)
        # On camera cut, reset persistence so tracker starts fresh IDs
        persist = not is_camera_cut

        try:
            results = self.model.track(
                frame,
                conf=self.conf,
                iou=self.iou,
                tracker=self.tracker_config,
                persist=persist,
                verbose=False,
            )
        except Exception as e:
            # Fallback to detection without tracking if tracker fails
            results = self.model(frame, conf=self.conf, verbose=False)

        if not results or len(results) == 0:
            objects = []
        else:
            objects = self._parse_results(results[0])

        # Fallback puck-only YOLO call when the primary missed the puck.
        # Runs on a SEPARATE model instance because predict() corrupts the
        # primary tracker's ByteTrack state. Throttled to every other
        # missed frame: the PuckFilter coast window already covers 5
        # frames, so an alternating fallback still recovers dropouts
        # without paying double inference cost on every frame.
        has_puck = any(o.class_name == "puck" for o in objects)
        if not has_puck and self._puck_class_id is not None and (self._frame_count % 2 == 0):
            try:
                fb = self.fallback_model.predict(
                    frame,
                    classes=[self._puck_class_id],
                    conf=self.puck_fallback_conf,
                    verbose=False,
                )
                if fb and len(fb) > 0:
                    fb_objects = self._parse_results(fb[0])
                    objects.extend(o for o in fb_objects if o.class_name == "puck")
            except Exception:
                pass  # Fallback is best-effort; never crash the pipeline

        # Single-puck enforcement + on-ice filter + short coast.
        # On camera cut, reset coast state so we don't synthesize a ghost
        # at a position from a totally different angle.
        if is_camera_cut:
            self.puck_filter._last_pos = None
            self.puck_filter._frames_since_seen = 9999

        objects = self.puck_filter.filter(objects, frame)
        return objects

    def _parse_results(self, result) -> list:
        """Convert ultralytics result to list of TrackedObject."""
        objects = []
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return objects

        for i in range(len(boxes)):
            # Get class info
            class_id = int(boxes.cls[i].cpu().item())
            class_name = self.class_names.get(class_id, f"unknown_{class_id}")
            conf = float(boxes.conf[i].cpu().item())

            # Get bounding box
            xyxy = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])

            # Get track ID (may not exist if tracking failed)
            track_id = -1
            if boxes.id is not None:
                track_id = int(boxes.id[i].cpu().item())

            # Compute center
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            obj = TrackedObject(
                track_id=track_id,
                class_name=class_name,
                class_id=class_id,
                bbox=(x1, y1, x2, y2),
                center=(cx, cy),
                confidence=conf,
            )
            objects.append(obj)

        return objects

    def reset(self):
        """Reset all tracking state."""
        self._frame_count = 0

    @staticmethod
    def filter_by_class(objects: list, *class_names: str) -> list:
        """Filter tracked objects by class name(s)."""
        return [o for o in objects if o.class_name in class_names]

    @staticmethod
    def get_puck(objects: list) -> Optional[TrackedObject]:
        """Get highest-confidence puck detection, or None."""
        pucks = [o for o in objects if o.class_name == "puck"]
        if not pucks:
            return None
        return max(pucks, key=lambda p: p.confidence)

    @staticmethod
    def get_players(objects: list) -> list:
        """Get all player detections."""
        return [o for o in objects if o.class_name == "player"]

    @staticmethod
    def get_goalies(objects: list) -> list:
        """Get all goalie detections."""
        return [o for o in objects if o.class_name == "goalie"]

    @staticmethod
    def get_rink_landmarks(objects: list) -> dict:
        """Get rink landmarks grouped by class name."""
        landmarks = {}
        for o in objects:
            if o.class_name in RINK_LANDMARK_CLASSES:
                if o.class_name not in landmarks:
                    landmarks[o.class_name] = []
                landmarks[o.class_name].append(o)
        return landmarks
