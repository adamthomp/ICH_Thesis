
#!/usr/bin/env python3
"""
laser_centroid_calib.py

Finds the centroid of a laser spot in images and maps it into a calibrated
(X, Y) coordinate frame using a homography computed from 20 clicked points.

Assumptions
-----------
- All images are the same size and orientation (default example: 1120x920).
- The laser spot appears near a specific RGB color (default: 250,250,250) ± tol.
- Background may also be bright. We therefore select a connected component
  subject to area constraints to isolate the laser blob.
- Calibration: You will click 20 points in the image in the SAME ORDER as the
  provided world coordinates. A 3x3 homography is computed (pixel -> world).
- World coordinates for the 20 points are embedded below by default, but you
  can also provide a CSV/TSV with 20 rows of "X Y" or "X,Y".

Usage
-----
1) Calibrate on an image (uses the first image in --images_dir for display):
   python laser_centroid_calib.py --images_dir /path/to/images --calibrate \
       --calib_out /path/to/calibration.npz

   Optional: use a custom points file
       --points_file /path/to/points.csv

2) Process images (requires a saved calibration.npz to map to world coords):
   python laser_centroid_calib.py --images_dir /path/to/images \
       --calib_in /path/to/calibration.npz --results_csv /path/to/results.csv

Common options:
   --tol 10                # color tolerance around target RGB (±tol)
   --color 250,250,250     # target RGB color of the laser
   --min_area 10           # minimum area (pixels) for a component
   --max_area_frac 0.01    # max connected-component area as fraction of image
   --ext "png,jpg,jpeg"    # file extensions to include

Outputs
-------
- Calibration file (NPZ): homography H, clicked pixel points, world points.
- Results CSV: filename, px_x, px_y, world_X, world_Y, area.
"""

import argparse
import csv
import glob
import os
import sys
from typing import List, Tuple, Optional

import numpy as np
import cv2
import matplotlib.pyplot as plt

# --------------------------- Default world points (20x2) ---------------------------
# Provided by the user, in the order they will be clicked in the image.
DEFAULT_WORLD_POINTS = np.array([
    [ 34.1895724 , 112.2947278 ],
    [ -8.846660442,  58.29168078],
    [ -44.2980125 , 146.1256879 ],
    [ -6.223012286, 167.0642373 ],
    [ -29.55771497,  90.00696566],
    [ -67.24044772, 123.0712913 ],
    [ 45.37136174, 199.3113009 ],
    [ -45.46746707, 118.3758179 ],
    [ -0.362311017,  82.19487354],
    [ 19.81897864,  75.43023854],
    [ -45.74437922, 173.2836104 ],
    [ 76.80060625, 164.5579778 ],
    [ -72.23192268, 143.3872552 ],
    [ 48.57564843,  88.84331783],
    [ 41.99283715,  51.78027677],
    [ 12.85236309, 132.4313666 ],
    [ -71.02956959, 192.6766381 ],
    [ 37.51407115, 146.7406027 ],
    [ 67.68066378, 182.4138486 ],
    [ -61.69999415,  96.22438023]
], dtype=np.float64)


# --------------------------- Utilities ---------------------------

def parse_color(s: str) -> Tuple[int, int, int]:
    """Parse 'R,G,B' into a tuple of ints."""
    parts = s.split(',')
    if len(parts) != 3:
        raise ValueError("Color must be in 'R,G,B' format")
    r, g, b = (int(p.strip()) for p in parts)
    for v in (r, g, b):
        if not (0 <= v <= 255):
            raise ValueError("Each color channel must be in [0,255]")
    return (r, g, b)


def list_images(images_dir: str, ext_list: List[str]) -> List[str]:
    files = []
    for ext in ext_list:
        files.extend(glob.glob(os.path.join(images_dir, f'**/*.{ext}'), recursive=True))
        files.extend(glob.glob(os.path.join(images_dir, f'*.{ext}')))
    files = sorted(set(files))
    return files


def load_world_points(points_file: Optional[str]) -> np.ndarray:
    """Load world points from a file or return default if None.
    Accepts CSV/TSV/space-delimited with two columns (X, Y)."""
    if points_file is None:
        return DEFAULT_WORLD_POINTS.copy()

    pts = []
    with open(points_file, 'r', newline='') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Replace commas and tabs by spaces to normalize
            for ch in [',', '\t']:
                line = line.replace(ch, ' ')
            parts = [p for p in line.split(' ') if p]
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
                pts.append([x, y])
            except ValueError:
                continue
    pts = np.array(pts, dtype=np.float64)
    if pts.shape != (20, 2):
        raise ValueError(f"Expected 20 world points, got shape {pts.shape}.")
    return pts


# --------------------------- Calibration (click collection) ---------------------------

def collect_calibration_clicks(img_bgr: np.ndarray, n_points: int = 20) -> np.ndarray:
    """Collect n_points pixel coordinates via interactive clicking using matplotlib.
    Returns an array (n_points, 2) of (x, y) pixel coordinates.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.title(f"Click {n_points} points in order. Close window when done.", fontsize=12)
    plt.axis('on')

    clicked = []

    # Click one-by-one so we can annotate indices live
    for i in range(n_points):
        plt.title(f"Click point {i+1}/{n_points} (in order).", fontsize=12)
        pts = plt.ginput(1, timeout=0)  # block until a click
        if not pts:
            print("No point clicked; aborting.")
            plt.close()
            sys.exit(1)
        x, y = pts[0]
        clicked.append([x, y])
        plt.scatter([x], [y], marker='o', s=50)
        plt.text(x+5, y+5, str(i+1), color='yellow', fontsize=10)
        plt.draw()

    plt.title("All points captured. Close the window to continue.", fontsize=12)
    plt.show(block=True)
    return np.array(clicked, dtype=np.float64)


def compute_homography(pixel_pts: np.ndarray, world_pts: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute homography H that maps pixel_pts (Nx2) to world_pts (Nx2).
    Returns H (3x3) and RMSE of reprojection (in world units)."""
    if pixel_pts.shape != world_pts.shape or pixel_pts.shape[1] != 2:
        raise ValueError("pixel_pts and world_pts must both be Nx2 with same N.")
    H, mask = cv2.findHomography(pixel_pts, world_pts, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if H is None:
        raise RuntimeError("cv2.findHomography failed to compute a valid homography.")

    # Compute RMSE of mapping pixel_pts -> world using H
    ones = np.ones((pixel_pts.shape[0], 1))
    pix_h = np.hstack([pixel_pts, ones])  # Nx3
    world_h = (H @ pix_h.T).T             # Nx3
    world_xy = world_h[:, :2] / world_h[:, [2]]

    diffs = world_xy - world_pts
    rmse = float(np.sqrt(np.mean(np.sum(diffs**2, axis=1))))
    return H, rmse


def save_calibration(path: str, H: np.ndarray, pixel_pts: np.ndarray, world_pts: np.ndarray, image_shape: Tuple[int, int, int]):
    np.savez_compressed(path, H=H, pixel_pts=pixel_pts, world_pts=world_pts, image_shape=np.array(image_shape))
    print(f"[OK] Saved calibration -> {path}")


def load_calibration(path: str) -> np.ndarray:
    data = np.load(path)
    H = data['H']
    return H


def pixel_to_world(pt_xy: Tuple[float, float], H: np.ndarray) -> Tuple[float, float]:
    x, y = pt_xy
    vec = np.array([x, y, 1.0], dtype=np.float64)
    w = H @ vec
    X = w[0] / w[2]
    Y = w[1] / w[2]
    return float(X), float(Y)


# --------------------------- Laser centroid detection ---------------------------

def find_laser_centroid(
    img_bgr: np.ndarray,
    target_rgb: Tuple[int, int, int] = (250, 250, 250),
    tol: int = 10,
    min_area: int = 10,
    max_area_frac: float = 0.01
) -> Tuple[Optional[Tuple[float, float]], int]:
    """
    Returns:
        (centroid_xy, area_pixels)
        centroid_xy: (x, y) in pixel coordinates or None if not found.
        area_pixels: area of the selected connected component (0 if none).
    """
    # Convert to RGB for comparison with target_rgb
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Build inRange mask for target color ± tol
    lower = np.array([max(0, c - tol) for c in target_rgb], dtype=np.uint8)
    upper = np.array([min(255, c + tol) for c in target_rgb], dtype=np.uint8)
    mask = cv2.inRange(img_rgb, lower, upper)

    # Morphological open/close to remove specks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    h, w = mask.shape[:2]
    max_area = int(max_area_frac * (h * w))

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Select component by area constraints; pick the largest that is <= max_area
    best_idx = -1
    best_area = -1
    for label in range(1, num_labels):  # skip background=0
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        if area > best_area:
            best_area = area
            best_idx = label

    if best_idx == -1:
        # Fallback: if everything was too big/small, try picking the largest non-background
        # but guard against entire page selection by hard cap of 25% image area
        hard_cap = int(0.25 * (h * w))
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area <= hard_cap and area > best_area:
                best_area = area
                best_idx = label

    if best_idx == -1 or best_area <= 0:
        return None, 0

    cx, cy = centroids[best_idx]
    return (float(cx), float(cy)), int(best_area)


# --------------------------- Main pipeline ---------------------------

def process_images(
    images_dir: str,
    H: Optional[np.ndarray],
    target_rgb: Tuple[int, int, int],
    tol: int,
    min_area: int,
    max_area_frac: float,
    ext_list: List[str]
) -> List[Tuple[str, Optional[float], Optional[float], Optional[float], Optional[float], int]]:
    files = list_images(images_dir, ext_list)
    if not files:
        print(f"[WARN] No images found in {images_dir} with extensions {ext_list}")
        return []

    results = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Failed to read image: {f}")
            results.append((f, None, None, None, None, 0))
            continue

        centroid_px, area = find_laser_centroid(
            img, target_rgb=target_rgb, tol=tol, min_area=min_area, max_area_frac=max_area_frac
        )

        if centroid_px is None:
            results.append((f, None, None, None, None, 0))
            continue

        px_x, px_y = centroid_px
        if H is not None:
            world_X, world_Y = pixel_to_world((px_x, px_y), H)
        else:
            world_X, world_Y = None, None

        results.append((f, px_x, px_y, world_X, world_Y, area))

    return results


def write_results_csv(path: str, rows: List[Tuple[str, Optional[float], Optional[float], Optional[float], Optional[float], int]]):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "px_x", "px_y", "world_X", "world_Y", "area_px"])
        for row in rows:
            writer.writerow(row)
    print(f"[OK] Wrote results -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Laser centroid finder with 20-point homography calibration.")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing images to process.")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration by clicking 20 points on the first image.")
    parser.add_argument("--calib_out", type=str, default=None, help="Path to save calibration (NPZ). Required if --calibrate.")
    parser.add_argument("--calib_in", type=str, default=None, help="Path to load an existing calibration (NPZ).")
    parser.add_argument("--points_file", type=str, default=None, help="Optional file with 20 world points (X,Y).")
    parser.add_argument("--results_csv", type=str, default=None, help="CSV path for results (defaults to images_dir/results.csv).")
    parser.add_argument("--tol", type=int, default=10, help="Color tolerance around target RGB (± tol).")
    parser.add_argument("--color", type=str, default="250,250,250", help="Target RGB color as 'R,G,B'.")
    parser.add_argument("--min_area", type=int, default=10, help="Minimum area (pixels) for a valid connected component.")
    parser.add_argument("--max_area_frac", type=float, default=0.01, help="Maximum area as a fraction of image size (to avoid selecting the whole page).")
    parser.add_argument("--ext", type=str, default="png,jpg,jpeg,bmp,tif,tiff", help="Comma-separated list of image extensions to include.")
    args = parser.parse_args()

    target_rgb = parse_color(args.color)
    ext_list = [e.strip().lower() for e in args.ext.split(',') if e.strip()]

    # List images
    files = list_images(args.images_dir, ext_list)
    if not files:
        print(f"[ERROR] No images found in {args.images_dir} with extensions {ext_list}")
        sys.exit(1)

    # Load world points
    world_pts = load_world_points(args.points_file)

    # Calibration path defaults
    if args.calibrate and args.calib_out is None:
        args.calib_out = os.path.join(args.images_dir, "calibration.npz")

    H = None

    if args.calibrate:
        first_img = cv2.imread(files[0], cv2.IMREAD_COLOR)
        if first_img is None:
            print(f"[ERROR] Could not read first image for calibration: {files[0]}")
            sys.exit(1)
        print("[INFO] Calibration: Click the 20 points in the SAME ORDER as the provided world coordinates.")
        print("       A marker and index will be drawn after each click.")
        pixel_pts = collect_calibration_clicks(first_img, n_points=20)
        H, rmse = compute_homography(pixel_pts, world_pts)
        print(f"[OK] Homography computed. Reprojection RMSE: {rmse:.3f} (world units)")
        if args.calib_out:
            save_calibration(args.calib_out, H, pixel_pts, world_pts, first_img.shape)

    # Load calibration if provided
    if args.calib_in is not None and H is None:
        H = load_calibration(args.calib_in)
        print(f"[OK] Loaded homography from {args.calib_in}")

    # Where to write results
    results_csv = args.results_csv or os.path.join(args.images_dir, "results.csv")

    # Process all images
    rows = process_images(
        args.images_dir, H, target_rgb, args.tol, args.min_area, args.max_area_frac, ext_list
    )

    # Write CSV
    write_results_csv(results_csv, rows)


if __name__ == "__main__":
    main()
