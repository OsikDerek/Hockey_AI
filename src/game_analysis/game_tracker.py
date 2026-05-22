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
        landmark_model_path: str = "models/landmarks_yolov8n.pt",
        landmark_conf: float = 0.20,
        puck_imgsz: int = 1920,
        puck_conf: float = 0.15,
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
        # Puck detection is a dedicated high-resolution pass. The puck is a
        # tiny object (a few pixels on a 1080p broadcast frame); the primary
        # track() call at the default imgsz downscales it into near
        # invisibility. A puck-only predict() at native broadcast resolution
        # lifts raw per-frame detection from ~36% to ~87% on caufield_goal.
        # The PuckFilter still rejects bad candidates via on-ice + proximity.
        self.puck_imgsz = puck_imgsz
        self.puck_conf = puck_conf

        # Optional rink-landmarks specialist model. When present, runs in
        # parallel with the main model and its centroid / faceoff / goal
        # detections take precedence (the main model under-detects these
        # because its representation budget is dominated by the "player"
        # class). Falls back gracefully when the file isn't present.
        self.landmark_model = None
        self.landmark_conf = landmark_conf
        landmark_file = Path(landmark_model_path)
        if landmark_file.is_file():
            try:
                self.landmark_model = YOLO(str(landmark_file))
                print(f"  GameTracker: landmark specialist loaded from {landmark_file} "
                      f"(classes: {self.landmark_model.names})")
            except Exception as e:
                print(f"  GameTracker: landmark specialist failed to load: {e}")
                self.landmark_model = None

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

        # Single-puck enforcement + on-ice filtering + short coast on dropouts.
        # min_conf tracks the puck pass conf so PuckFilter never re-discards
        # a detection the high-res pass already accepted.
        self.puck_filter = PuckFilter(min_conf=min(puck_conf, 0.15))

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

        # Dedicated high-resolution puck pass. Runs every frame on a SEPARATE
        # model instance (predict() corrupts the primary tracker's ByteTrack
        # state) at native broadcast resolution, puck class only. Puck
        # detections from the primary track() call are discarded in favour of
        # this pass — at the default imgsz the primary barely sees the puck.
        objects = [o for o in objects if o.class_name != "puck"]
        if self._puck_class_id is not None:
            try:
                fb = self.fallback_model.predict(
                    frame,
                    classes=[self._puck_class_id],
                    conf=self.puck_conf,
                    imgsz=self.puck_imgsz,
                    verbose=False,
                )
                if fb and len(fb) > 0:
                    fb_objects = self._parse_results(fb[0])
                    objects.extend(o for o in fb_objects if o.class_name == "puck")
            except Exception:
                pass  # Best-effort; never crash the pipeline

        # Landmark specialist: when loaded, run alongside the main model
        # to augment rink-landmark coverage. We UNION both sets of
        # detections (deduping near-coincident centers within ~30 px) —
        # specialist may catch landmarks the main model misses and vice
        # versa, especially across different broadcast styles. Player /
        # puck / goalie / referee classes still come from the main
        # HockeyAI model only.
        if self.landmark_model is not None:
            try:
                lm_results = self.landmark_model.predict(
                    frame, conf=self.landmark_conf, verbose=False,
                )
                if lm_results and len(lm_results) > 0:
                    landmark_objs = self._parse_landmark_results(lm_results[0])
                    objects = self._merge_landmark_detections(objects, landmark_objs)
            except Exception:
                # Best-effort: never break the main pipeline if the
                # specialist crashes mid-frame.
                pass

        # Single-puck enforcement + on-ice filter + short coast.
        # On camera cut, reset coast state so we don't synthesize a ghost
        # at a position from a totally different angle.
        if is_camera_cut:
            self.puck_filter._last_pos = None
            self.puck_filter._frames_since_seen = 9999

        objects = self.puck_filter.filter(objects, frame)
        return objects

    @staticmethod
    def _merge_landmark_detections(existing: list, specialist: list,
                                    dedup_radius_px: float = 30.0) -> list:
        """Union main-model + specialist detections, dropping near-duplicates.

        For each specialist detection, drop it if the existing list already
        contains a same-class detection within `dedup_radius_px`. Keeps the
        higher-confidence detection in case of ties (specialist wins).
        """
        if not specialist:
            return existing
        landmark_classes = {"centroid", "faceoff", "goal"}
        out = list(existing)
        for new_obj in specialist:
            if new_obj.class_name not in landmark_classes:
                continue
            keep = True
            for i, ex in enumerate(out):
                if ex.class_name != new_obj.class_name:
                    continue
                dx = ex.center[0] - new_obj.center[0]
                dy = ex.center[1] - new_obj.center[1]
                if dx * dx + dy * dy < dedup_radius_px * dedup_radius_px:
                    if new_obj.confidence > ex.confidence:
                        out[i] = new_obj  # specialist replaces lower-conf existing
                    keep = False
                    break
            if keep:
                out.append(new_obj)
        return out

    def _parse_landmark_results(self, result) -> list:
        """Parse landmark-specialist detections.

        The specialist's class indices are 0=centroid, 1=faceoff, 2=goal
        (preserved from the SHL training data). No track_id since the
        specialist runs as a per-frame predict (no ByteTrack state).
        """
        out = []
        if result is None or result.boxes is None:
            return out
        boxes = result.boxes
        names = self.landmark_model.names if self.landmark_model else {}
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            class_name = names.get(cls_id, f"unknown_{cls_id}")
            # Normalize the typo from the original model
            if class_name == "centriod":
                class_name = "centroid"
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            obj = TrackedObject(
                track_id=-1,  # specialist has no tracker; landmarks don't need stable IDs
                class_name=class_name,
                class_id=cls_id,
                bbox=(x1, y1, x2, y2),
                center=((x1 + x2) / 2, (y1 + y2) / 2),
                confidence=conf,
            )
            out.append(obj)
        return out

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
            # The HockeyAI weights ship with the class label "centriod"
            # (typo). Normalize at parse time so downstream code can use
            # the correct spelling.
            if class_name == "centriod":
                class_name = "centroid"
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
