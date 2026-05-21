"""Derive the 56-keypoint rink template from the HockeyRink annotations.

The HockeyRink model detects 56 rink keypoints but the keypoint->ice
mapping was never published. The 56 points are coplanar (rink markings),
so across the 661 annotated frames they're related by homographies.

Method:
  1. Parse every annotation -> 56 keypoints in pixel coords per frame
     (visibility >= 1, i.e. annotator-marked even if occluded).
  2. Seed the template with the frame that has the most keypoints.
  3. Iteratively merge every other frame: fit a homography from that
     frame to the current template using shared keypoints, project its
     keypoints into the template plane, add any not yet known + refine
     existing ones by averaging.
  4. Result: all 56 keypoints in one consistent plane.

Output:
  output/_rink_template/template_raw.json  — 56 pts in seed-frame plane
  output/_rink_template/template_render.png — indexed render to eyeball

A later anchoring step maps this projective template to NHL ice feet.

Run:  .venv/Scripts/python.exe scripts/reconstruct_rink_template.py
"""
from __future__ import annotations
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANN_DIR = PROJECT_ROOT / "data/hockeyrink_meta/Dataset/SHL/annotations"
OUT_DIR = PROJECT_ROOT / "output/_rink_template"
FRAME_W, FRAME_H = 1920, 1080
NUM_KP = 56


def load_annotations():
    """Return list of (name, kp array (56,3) [x_px,y_px,vis])."""
    frames = []
    for path in sorted(glob.glob(str(ANN_DIR / "*.txt"))):
        vals = []
        with open(path) as f:
            line = f.readline().split()
        if len(vals := [float(v) for v in line]) < 5 + NUM_KP * 3:
            continue
        kp = np.zeros((NUM_KP, 3))
        for i in range(NUM_KP):
            x, y, v = vals[5 + i * 3: 5 + i * 3 + 3]
            kp[i] = [x * FRAME_W, y * FRAME_H, v]
        frames.append((Path(path).stem, kp))
    return frames


def _grow_from_seed(frames, seed_idx):
    """Single greedy merge pass from a seed frame. Returns template
    dict {i: list of (x,y) estimates} and the set of merged frame idxs."""
    seed_kp = frames[seed_idx][1]
    template = {i: [] for i in range(NUM_KP)}
    for i in range(NUM_KP):
        if seed_kp[i, 2] >= 1:
            template[i].append(seed_kp[i, :2].copy())

    def tmpl_pt(i):
        return np.mean(template[i], axis=0) if template[i] else None

    merged = {seed_idx}
    for _ in range(6):
        progress = False
        for fi, (_name, kp) in enumerate(frames):
            if fi in merged:
                continue
            shared = [i for i in range(NUM_KP) if kp[i, 2] >= 1 and template[i]]
            if len(shared) < 10:
                continue
            src = np.array([kp[i, :2] for i in shared], dtype=np.float64)
            dst = np.array([tmpl_pt(i) for i in shared], dtype=np.float64)
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if H is None or mask is None or int(mask.sum()) < 10:
                continue
            proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
            if np.median(np.linalg.norm(proj - dst, axis=1)) > 8.0:
                continue
            marked = [i for i in range(NUM_KP) if kp[i, 2] >= 1]
            mp = np.array([kp[i, :2] for i in marked], dtype=np.float64)
            mp_t = cv2.perspectiveTransform(mp.reshape(-1, 1, 2), H).reshape(-1, 2)
            for i, p in zip(marked, mp_t):
                template[i].append(p)
            merged.add(fi)
            progress = True
        if not progress:
            break
    return template, merged


def _robust_avg(template):
    final = {}
    for i in range(NUM_KP):
        if not template[i]:
            continue
        pts = np.array(template[i])
        med = np.median(pts, axis=0)
        d = np.linalg.norm(pts - med, axis=1)
        mad = np.median(d) + 1e-6
        keep = pts[d < max(15.0, 2.5 * mad)]
        final[i] = np.mean(keep, axis=0) if len(keep) else med
    return final


def _stitch(main, cluster):
    """Merge a second cluster (dict i->xy) into the main template plane
    via a homography fit on shared keypoints. Returns updated main."""
    shared = [i for i in cluster if i in main]
    if len(shared) < 4:
        return main, 0
    src = np.array([cluster[i] for i in shared], dtype=np.float64)
    dst = np.array([main[i] for i in shared], dtype=np.float64)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
    if H is None:
        return main, 0
    added = 0
    for i, p in cluster.items():
        if i in main:
            continue
        pt = cv2.perspectiveTransform(
            np.array([[p]], dtype=np.float64), H).reshape(2)
        main[i] = pt
        added += 1
    return main, added


def reconstruct(frames):
    """Reconstruct all 56 keypoints in one plane via multi-cluster
    growth + stitching, then refine.

    A single seed only reaches the rink end it sees. We grow clusters
    from multiple seeds and stitch them together through shared
    keypoints, then re-fit every frame against the unified template.
    """
    template = None
    used_seeds = set()
    for _cluster in range(6):
        # Seed = unmerged frame maximizing keypoints NOT yet in template.
        def novelty(i):
            kp = frames[i][1]
            marked = {k for k in range(NUM_KP) if kp[k, 2] >= 1}
            if template is None:
                return len(marked)
            return len(marked - set(template))
        cand = [i for i in range(len(frames)) if i not in used_seeds]
        if not cand:
            break
        seed_idx = max(cand, key=novelty)
        if novelty(seed_idx) < 4:
            break
        used_seeds.add(seed_idx)
        cl_lists, merged = _grow_from_seed(frames, seed_idx)
        cluster = _robust_avg(cl_lists)
        if template is None:
            template = cluster
            print(f"  cluster 1: seed {frames[seed_idx][0][:8]}, "
                  f"{len(merged)} frames, {len(template)}/{NUM_KP} keypoints")
        else:
            before = len(template)
            template, added = _stitch(template, cluster)
            print(f"  cluster {_cluster+1}: seed {frames[seed_idx][0][:8]}, "
                  f"+{added} keypoints -> {len(template)}/{NUM_KP}")
        if len(template) >= NUM_KP:
            break
    merged = set()  # recomputed below isn't needed; placeholder
    print(f"  pre-refine: {len(template)}/{NUM_KP} keypoints")

    # Refinement passes: re-fit every frame against the full template.
    for rp in range(4):
        acc = {i: [] for i in range(NUM_KP)}
        n_fit = 0
        total_err = []
        for _name, kp in frames:
            shared = [i for i in range(NUM_KP) if kp[i, 2] >= 1 and i in template]
            if len(shared) < 10:
                continue
            src = np.array([kp[i, :2] for i in shared], dtype=np.float64)
            dst = np.array([template[i] for i in shared], dtype=np.float64)
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if H is None or mask is None or int(mask.sum()) < 10:
                continue
            proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
            err = np.median(np.linalg.norm(proj - dst, axis=1))
            if err > 8.0:
                continue
            total_err.append(err)
            marked = [i for i in range(NUM_KP) if kp[i, 2] >= 1]
            mp = np.array([kp[i, :2] for i in marked], dtype=np.float64)
            mp_t = cv2.perspectiveTransform(mp.reshape(-1, 1, 2), H).reshape(-1, 2)
            for i, p in zip(marked, mp_t):
                acc[i].append(p)
            n_fit += 1
        template = _robust_avg(acc)
        me = float(np.median(total_err)) if total_err else -1
        print(f"  refine {rp+1}: {n_fit} frames fit, {len(template)}/{NUM_KP} "
              f"keypoints, median reproj err {me:.2f}px")

    return template, len(merged), len(frames)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = load_annotations()
    print(f"loaded {len(frames)} annotated frames")
    if not frames:
        print("no annotations found"); return 2

    template, n_merged, n_total = reconstruct(frames)
    print(f"\nreconstructed {len(template)}/{NUM_KP} keypoints "
          f"from {n_merged}/{n_total} frames")

    # Save raw template
    raw = {str(i): [float(p[0]), float(p[1])] for i, p in template.items()}
    (OUT_DIR / "template_raw.json").write_text(json.dumps(raw, indent=1))

    # Render: normalize to a canvas, draw indexed dots
    pts = np.array(list(template.values()))
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    span = mx - mn
    W, H = 1700, int(1700 * span[1] / span[0]) + 120
    canvas = np.full((H, W, 3), 30, np.uint8)
    for i, p in template.items():
        nx = int(60 + (p[0] - mn[0]) / span[0] * (W - 120))
        ny = int(60 + (p[1] - mn[1]) / span[1] * (H - 120))
        cv2.circle(canvas, (nx, ny), 7, (0, 230, 0), -1)
        cv2.putText(canvas, str(i), (nx + 9, ny + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(str(OUT_DIR / "template_render.png"), canvas)
    print(f"wrote {OUT_DIR}/template_raw.json + template_render.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
