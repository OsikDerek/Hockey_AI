"""Synthetic training-data generator for the learned rink-registration model.

Real labeled hockey-rink-keypoint data is scarce, so we follow the
Shang et al. ("Rink-Agnostic Hockey Rink Registration") strategy:
train primarily on synthetic data, then domain-adapt on real frames.

This module renders the NHL rink at thousands of physically-plausible
broadcast-camera viewpoints via a pinhole camera model. For each sample
we KNOW the exact ice->image homography, so every rink keypoint's pixel
position and visibility is ground truth — no manual labeling.

Output is YOLO-pose format (one .txt label per .jpg image):
    <class> <bbox_cx> <bbox_cy> <bbox_w> <bbox_h> <kp1_x> <kp1_y> <kp1_v> ...
all normalized to [0,1]; visibility v in {0 = absent, 2 = visible}.

Camera model
------------
The ice surface is the world z=0 plane (units = feet, NHL coords).
A pinhole camera at C looks at an ice point L. For points on z=0 the
projection collapses to a 3x3 homography  H = K [r1 r2 t]  where r1,r2
are the first two columns of the world->camera rotation and t its
translation. Broadcast cameras sit elevated on one side of the rink;
we sample C, L, and focal length within ranges that reproduce the
NHL elevated press-box look (incl. pan + zoom).
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import cv2
import numpy as np

from .keypoints import (
    KEYPOINT_ICE_XY, NUM_KEYPOINTS, RINK_LENGTH_FT, RINK_WIDTH_FT,
    CENTER_X, CENTER_Y, GOAL_LINE_X, BLUE_LINE_X, RED_LINE_X,
    EZ_DOT_X, NZ_DOT_X, DOT_Y, CIRCLE_RADIUS,
)

# Output image size — matches the broadcast clips (1280x720).
OUT_W, OUT_H = 1280, 720

# Top-down rink template resolution: pixels per foot.
TEMPLATE_PX_PER_FT = 12
TEMPLATE_W = int(RINK_LENGTH_FT * TEMPLATE_PX_PER_FT)   # 2400
TEMPLATE_H = int(RINK_WIDTH_FT * TEMPLATE_PX_PER_FT)    # 1020

CORNER_RADIUS_FT = 28.0

# Colors (BGR)
ICE = (235, 230, 222)
BLUE_LINE = (150, 70, 30)
RED_LINE = (40, 40, 200)
DARK = (60, 55, 50)


def _ft_to_template(x_ft: float, y_ft: float) -> tuple[int, int]:
    return (int(round(x_ft * TEMPLATE_PX_PER_FT)),
            int(round(y_ft * TEMPLATE_PX_PER_FT)))


def draw_rink_template() -> np.ndarray:
    """Render the NHL rink top-down (ice + all painted markings)."""
    img = np.full((TEMPLATE_H, TEMPLATE_W, 3), ICE, dtype=np.uint8)
    pf = TEMPLATE_PX_PER_FT

    def line(x1f, y1f, x2f, y2f, color, thick_ft):
        cv2.line(img, _ft_to_template(x1f, y1f), _ft_to_template(x2f, y2f),
                 color, max(1, int(thick_ft * pf)))

    def circle(xf, yf, rf, color, thick_ft, fill=False):
        c = _ft_to_template(xf, yf)
        r = int(rf * pf)
        cv2.circle(img, c, r, color, -1 if fill else max(1, int(thick_ft * pf)))

    # Vertical painted lines (run across the rink width)
    for bx in BLUE_LINE_X:
        line(bx, 0, bx, RINK_WIDTH_FT, BLUE_LINE, 1.0)
    line(RED_LINE_X, 0, RED_LINE_X, RINK_WIDTH_FT, RED_LINE, 1.0)
    for gx in GOAL_LINE_X:
        line(gx, 4, gx, RINK_WIDTH_FT - 4, RED_LINE, 0.2)

    # Centre circle + centre dot
    circle(CENTER_X, CENTER_Y, CIRCLE_RADIUS, BLUE_LINE, 0.2)
    circle(CENTER_X, CENTER_Y, 1.0, BLUE_LINE, 0, fill=True)

    # Faceoff circles + dots (end zones)
    for ex in EZ_DOT_X:
        for ey in DOT_Y:
            circle(ex, ey, CIRCLE_RADIUS, RED_LINE, 0.2)
            circle(ex, ey, 1.0, RED_LINE, 0, fill=True)
    # Neutral-zone dots (no circle)
    for nx in NZ_DOT_X:
        for ny in DOT_Y:
            circle(nx, ny, 1.0, RED_LINE, 0, fill=True)

    # Goal creases (rough) + goal mouths
    for gx, side in ((GOAL_LINE_X[0], 1), (GOAL_LINE_X[1], -1)):
        crease = _ft_to_template(gx, CENTER_Y)
        cv2.ellipse(img, crease, (int(6 * pf), int(4 * pf)),
                    0, -90 * side, 90 * side, BLUE_LINE, max(1, int(0.2 * pf)))

    # Boards: dark border with rounded corners (approx via thick rect outline)
    cv2.rectangle(img, (0, 0), (TEMPLATE_W - 1, TEMPLATE_H - 1),
                  DARK, max(2, int(0.5 * pf)))
    return img


def _look_at_homography(cam, look, focal, up=(0.0, 0.0, 1.0)):
    """Pinhole ice(z=0)->image homography for a camera at `cam` looking
    at ice point `look`. Returns 3x3 H mapping (X_ft, Y_ft, 1) -> pixels.
    """
    cam = np.array(cam, dtype=np.float64)
    look = np.array(look, dtype=np.float64)
    up = np.array(up, dtype=np.float64)

    forward = look - cam
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)  # OpenCV cam: +y is down

    # world->camera rotation: rows are camera axes expressed in world frame
    R = np.stack([right, down, forward], axis=0)
    t = -R @ cam

    K = np.array([[focal, 0, OUT_W / 2.0],
                  [0, focal, OUT_H / 2.0],
                  [0, 0, 1.0]], dtype=np.float64)

    # For ice-plane points (X, Y, 0): projection uses cols [r1, r2, t]
    Rt = np.stack([R[:, 0], R[:, 1], t], axis=1)  # 3x3
    H = K @ Rt
    return H


def sample_broadcast_homography(rng: random.Random) -> np.ndarray:
    """Sample a physically-plausible NHL-broadcast ice->image homography.

    The broadcast main camera is FAR from the rink — up in the press
    box, ~100-180 ft back from the near boards, elevated ~35-90 ft,
    using a long lens. That far-away + long-lens geometry is what gives
    the characteristic broadcast look (compressed perspective, a whole
    offensive/neutral zone visible at once). Sampling the camera too
    close produces an extreme zoom into a tiny patch of ice.

    Visible rink span W ~= 1280 * D / focal, where D is camera->ice
    distance. With D ~140 ft and focal ~1500 that's ~120 ft of rink
    in frame — a realistic broadcast shot.
    """
    side = 1.0 if rng.random() < 0.85 else -1.0  # mostly the standard side
    # Camera sits FAR beyond the boards, elevated.
    setback = rng.uniform(95, 185)               # ft beyond the near board
    cam_y = -setback if side > 0 else RINK_WIDTH_FT + setback
    cam_x = CENTER_X + rng.uniform(-55, 55)
    cam_z = rng.uniform(35, 95)
    # Look at a point on the ice — pans along the rink length.
    look_x = rng.uniform(40, RINK_LENGTH_FT - 40)
    look_y = CENTER_Y + rng.uniform(-10, 10)
    focal = rng.uniform(1150, 2500)              # long-lens zoom range

    H = _look_at_homography((cam_x, cam_y, cam_z), (look_x, look_y, 0.0), focal)
    return H


def _project(H: np.ndarray, pts_ft: np.ndarray) -> np.ndarray:
    """Project (N,2) ice-foot points through H -> (N,2) pixel points."""
    n = pts_ft.shape[0]
    homog = np.hstack([pts_ft, np.ones((n, 1))])
    proj = (H @ homog.T).T
    w = proj[:, 2:3]
    w[np.abs(w) < 1e-9] = 1e-9
    return proj[:, :2] / w


_TEMPLATE_CACHE: np.ndarray | None = None


def _get_template() -> np.ndarray:
    """Cached top-down rink template — it's static, so render once."""
    global _TEMPLATE_CACHE
    if _TEMPLATE_CACHE is None:
        _TEMPLATE_CACHE = draw_rink_template()
    return _TEMPLATE_CACHE


def generate_sample(rng: random.Random):
    """Render one synthetic broadcast frame. Returns (image_bgr,
    keypoints_pixels (NUM_KEYPOINTS,2), visibility (NUM_KEYPOINTS,)).
    """
    template = _get_template()

    # ice-foot -> template-pixel matrix
    S = np.array([[TEMPLATE_PX_PER_FT, 0, 0],
                  [0, TEMPLATE_PX_PER_FT, 0],
                  [0, 0, 1.0]])
    H_ice2img = sample_broadcast_homography(rng)
    # warp matrix maps template-pixels -> output-image: H_ice2img @ inv(S)
    H_tmpl2img = H_ice2img @ np.linalg.inv(S)

    # Background: arena-dark with mild noise (boards/crowd stand-in).
    bg = np.full((OUT_H, OUT_W, 3), 0, dtype=np.uint8)
    bg[:] = (rng.randint(20, 55), rng.randint(20, 55), rng.randint(25, 60))
    warped = cv2.warpPerspective(
        template, H_tmpl2img, (OUT_W, OUT_H),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT,
    )
    # Composite: warpPerspective with BORDER_TRANSPARENT leaves bg where
    # the rink doesn't project. Build a mask from a white template warp.
    mask = cv2.warpPerspective(
        np.full((TEMPLATE_H, TEMPLATE_W), 255, np.uint8),
        H_tmpl2img, (OUT_W, OUT_H), flags=cv2.INTER_NEAREST,
    )
    out = bg.copy()
    out[mask > 0] = warped[mask > 0]

    # Project keypoints
    kp_ice = np.array(KEYPOINT_ICE_XY, dtype=np.float64)
    kp_px = _project(H_ice2img, kp_ice)
    vis = np.full(NUM_KEYPOINTS, 2, dtype=np.int32)
    for i, (px, py) in enumerate(kp_px):
        if not (0 <= px < OUT_W and 0 <= py < OUT_H):
            vis[i] = 0
        # also drop keypoints projected from behind the camera
    # Behind-camera check: any kp whose homogeneous w was negative is invalid
    homog = np.hstack([kp_ice, np.ones((NUM_KEYPOINTS, 1))])
    w = (H_ice2img @ homog.T).T[:, 2]
    vis[w <= 0] = 0

    # --- augmentation: brightness, blur, noise ---
    if rng.random() < 0.7:
        beta = rng.uniform(-35, 35)
        alpha = rng.uniform(0.8, 1.2)
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    if rng.random() < 0.3:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)
    if rng.random() < 0.5:
        noise = np.random.randint(-12, 12, out.shape, dtype=np.int16)
        out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return out, kp_px, vis


def _yolo_pose_label(kp_px: np.ndarray, vis: np.ndarray) -> str | None:
    """Build a YOLO-pose label line. bbox = tight box around visible
    keypoints (the model treats the rink as one object). Returns None
    if too few keypoints are visible to be useful."""
    visible = kp_px[vis == 2]
    if len(visible) < 4:
        return None
    x1, y1 = visible.min(axis=0)
    x2, y2 = visible.max(axis=0)
    # pad bbox a little and clamp
    pad = 30
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(OUT_W, x2 + pad); y2 = min(OUT_H, y2 + pad)
    cx = (x1 + x2) / 2 / OUT_W
    cy = (y1 + y2) / 2 / OUT_H
    bw = (x2 - x1) / OUT_W
    bh = (y2 - y1) / OUT_H
    parts = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
    for (px, py), v in zip(kp_px, vis):
        if v == 2:
            parts.append(f"{px / OUT_W:.6f} {py / OUT_H:.6f} 2")
        else:
            parts.append("0 0 0")
    return " ".join(parts)


def generate_dataset(out_dir: str, n_train: int, n_val: int, seed: int = 0):
    """Write a YOLO-pose dataset (images + labels + data.yaml)."""
    root = Path(out_dir)
    for split, n in (("train", n_train), ("val", n_val)):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        rng = random.Random(seed + (0 if split == "train" else 999_999))
        written = 0
        attempts = 0
        while written < n and attempts < n * 4:
            attempts += 1
            img, kp_px, vis = generate_sample(rng)
            label = _yolo_pose_label(kp_px, vis)
            if label is None:
                continue
            stem = f"{split}_{written:06d}"
            cv2.imwrite(str(root / "images" / split / f"{stem}.jpg"), img)
            (root / "labels" / split / f"{stem}.txt").write_text(label + "\n")
            written += 1
        print(f"  {split}: wrote {written} samples ({attempts} attempts)")

    yaml = (
        f"# Synthetic NHL rink-registration keypoint dataset\n"
        f"path: {root.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"kpt_shape: [{NUM_KEYPOINTS}, 3]\n"
        f"names:\n  0: rink\n"
    )
    (root / "data.yaml").write_text(yaml)
    print(f"  data.yaml written ({NUM_KEYPOINTS} keypoints/object)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/synth_rink", help="dataset output dir")
    p.add_argument("--train", type=int, default=4000)
    p.add_argument("--val", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--preview", action="store_true",
                   help="just render 6 preview frames to output/_synth_preview/")
    args = p.parse_args()

    if args.preview:
        rng = random.Random(args.seed)
        out = Path("output/_synth_preview")
        out.mkdir(parents=True, exist_ok=True)
        for i in range(6):
            img, kp_px, vis = generate_sample(rng)
            for (px, py), v in zip(kp_px, vis):
                if v == 2:
                    cv2.circle(img, (int(px), int(py)), 5, (0, 255, 0), -1)
                    cv2.circle(img, (int(px), int(py)), 6, (0, 0, 0), 1)
            cv2.imwrite(str(out / f"preview_{i}.jpg"), img)
        print(f"wrote 6 previews to {out}")
    else:
        print(f"Generating synthetic rink dataset -> {args.out}")
        generate_dataset(args.out, args.train, args.val, args.seed)
