#!/usr/bin/env python3
"""
Screen-triggered camera capture (macOS-friendly)

- Opens your camera (tries indices until one works) at 1920x1080 @ 30fps.
- Monitors ONE pixel on your Mac screen via MSS.
- When that pixel is "green" (near RGB (100,212,19) within tolerance),
  it saves a PNG from the camera to OUTPUT_DIR with the pattern: Rand_log_image_##.png.
- After a capture, it requires the pixel to become "not-green" before re-arming.
- Polls the screen at 20 Hz (adjustable).
- Provides a calibration helper to pick coordinates by clicking on a full-screen screenshot.

Dependencies:
    pip install opencv-python mss

Permissions:
    macOS → System Settings → Privacy & Security → Screen Recording
    Grant your terminal/IDE permission so MSS can read the screen.

Keys:
    q  - quit
    c  - run on-screen calibration helper (click to choose a pixel; Esc to cancel)
    s  - manual save (captures a frame immediately)
"""

import os
import re
import glob
import time
import argparse
from datetime import datetime
from typing import Tuple, Optional

import cv2
import numpy as np

try:
    import mss
except ImportError as e:
    raise SystemExit("Missing dependency: mss. Install with: pip install mss")


# ----------------------- User-configurable defaults ----------------------- #
# Logical (snipping-tool) resolution and physical (Retina) pixel resolution:
LOGICAL_RES   = (1728, 1117)        # what your snipping tool reports
PHYSICAL_RES  = (3456, 2234)        # actual Retina pixel resolution
PIXEL_XY      = (1330, 430)         # coordinate from snipping tool (logical units)
TARGET_RGB    = (100, 212, 19)      # trigger color (sRGB)
TOLERANCE     = 10                   # per-channel tolerance
POLL_HZ       = 20                   # screen poll frequency
DEBOUNCE_SEC  = 0.20                 # wait after capture before re-arming
OUT_DIR       = '/Users/adamthompson/Desktop/Thesis/Testing_Kinematics/capture_images_no_comp_2'  # default output dir
FNAME_PREFIX  = "Rand_log_image_"    # naming: Rand_log_image_##.png
FNAME_PAD_MIN = 3                    # at least 2 digits of zero padding

# Desired camera properties:
CAM_WIDTH     = 1920
CAM_HEIGHT    = 1080
CAM_FPS       = 30
# ------------------------------------------------------------------------- #


def compute_scale(logical_res: Tuple[int,int], physical_res: Tuple[int,int]) -> Tuple[float,float]:
    """Compute x,y scale factors from logical (points) to physical (pixels)."""
    lx, ly = logical_res
    px, py = physical_res
    if lx == 0 or ly == 0:
        return 1.0, 1.0
    return px / lx, py / ly


def logical_to_physical(xy_logical: Tuple[int,int],
                        scale_xy: Tuple[float,float]) -> Tuple[int,int]:
    sx, sy = scale_xy
    x, y = xy_logical
    return int(round(x * sx)), int(round(y * sy))


def color_is_match(rgb: Tuple[int,int,int],
                   target: Tuple[int,int,int],
                   tol: int) -> bool:
    """Check if an RGB color is within +-tol per channel of target."""
    return all(abs(int(c) - int(t)) <= tol for c, t in zip(rgb, target))


def get_next_index(output_dir: str, prefix: str) -> int:
    """Scan existing files to continue numbering without overwriting."""
    pattern = os.path.join(output_dir, f"{prefix}*.png")
    existing = glob.glob(pattern)
    numbers = []
    for p in existing:
        base = os.path.basename(p)
        m = re.search(rf"^{re.escape(prefix)}(\d+)\.png$", base)
        if m:
            try:
                numbers.append(int(m.group(1)))
            except ValueError:
                pass
    return (max(numbers) + 1) if numbers else 1


def format_fname(prefix: str, idx: int, pad_min: int) -> str:
    width = max(pad_min, len(str(idx)))
    return f"{prefix}{idx:0{width}d}.png"


def open_camera_interactive(preferred_index: Optional[int]=None) -> cv2.VideoCapture:
    """
    Try to open the camera. If preferred_index is given, try that first.
    Otherwise, scan indices 0..10 until one yields frames.
    """
    indices = list(range(0, 11))
    if preferred_index is not None and preferred_index in indices:
        indices.remove(preferred_index)
        indices.insert(0, preferred_index)

    cap = None
    for i in indices:
        cap_try = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        # Try to set desired properties (these may not always apply):
        cap_try.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
        cap_try.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        cap_try.set(cv2.CAP_PROP_FPS,          CAM_FPS)

        ok, frame = cap_try.read()
        if ok and frame is not None:
            cap = cap_try
            print(f"[INFO] Opened camera at index {i} "
                  f"({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ "
                  f"{int(cap.get(cv2.CAP_PROP_FPS))}fps)")
            break
        cap_try.release()

    if cap is None or not cap.isOpened():
        raise SystemExit("ERROR: Could not open any camera. "
                         "Try plugging in your webcam and/or change the index in the code.")
    return cap


def sample_screen_pixel(sct: "mss.mss", px: Tuple[int,int]) -> Tuple[int,int,int]:
    """
    Sample a single pixel from the primary monitor using MSS.
    Returns (R,G,B).
    """
    left, top = px
    # MSS expects the monitor dict or a rect. We'll use the primary monitor bounding box origin.
    mon = sct.monitors[1]  # primary
    # Ensure the requested pixel is within screen bounds:
    x = np.clip(left, mon["left"], mon["left"] + mon["width"] - 1)
    y = np.clip(top,  mon["top"],  mon["top"]  + mon["height"] - 1)
    rect = {"left": int(x), "top": int(y), "width": 1, "height": 1}
    img = np.asarray(sct.grab(rect))  # BGRA
    b, g, r, a = img[0, 0]
    return int(r), int(g), int(b)


def calibration_helper(sct: "mss.mss",
                       logical_res: Tuple[int,int],
                       physical_res: Tuple[int,int]) -> Tuple[int,int]:
    """
    Show a full-screen screenshot (scaled to fit window if needed).
    Let user click to choose a pixel; returns LOGICAL coordinates
    (so they match snipping-tool style numbers).
    Press ESC to cancel.
    """
    mon = sct.monitors[1]
    full = np.asarray(sct.grab(mon))[:, :, :3]  # BGR
    full = cv2.cvtColor(full, cv2.COLOR_BGRA2BGR)

    # Resize for display if too large for screen
    display_w = 1600
    scale_display = min(1.0, display_w / full.shape[1])
    disp = cv2.resize(full, (int(full.shape[1]*scale_display), int(full.shape[0]*scale_display)),
                      interpolation=cv2.INTER_AREA)

    clicked = {"pt": None}

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Map from display to physical pixels
            phys_x = int(round(x / scale_display))
            phys_y = int(round(y / scale_display))
            # Convert physical to logical coords:
            sx, sy = compute_scale(logical_res, physical_res)
            # sx, sy are px per logical; to get logical: divide
            log_x = int(round(phys_x / sx))
            log_y = int(round(phys_y / sy))
            clicked["pt"] = (log_x, log_y)

    win = "Calibration - click target pixel (ESC to cancel)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, disp)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        vis = disp.copy()
        # Show mouse position and color under cursor:
        mx, my = cv2.getWindowImageRect(win)[:2]  # window pos (not mouse)
        cv2.imshow(win, vis)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:  # ESC
            cv2.destroyWindow(win)
            return None
        if clicked["pt"] is not None:
            pt = clicked["pt"]
            cv2.destroyWindow(win)
            return pt


def main():
    parser = argparse.ArgumentParser(description="Screen-triggered camera capture")
    parser.add_argument("--out", default=OUT_DIR, help="Output directory for saved PNGs")
    parser.add_argument("--x", type=int, default=PIXEL_XY[0], help="Logical X (snipping-tool units)")
    parser.add_argument("--y", type=int, default=PIXEL_XY[1], help="Logical Y (snipping-tool units)")
    parser.add_argument("--tol", type=int, default=TOLERANCE, help="Per-channel tolerance")
    parser.add_argument("--poll", type=float, default=POLL_HZ, help="Screen poll frequency (Hz)")
    parser.add_argument("--debounce", type=float, default=DEBOUNCE_SEC, help="Debounce after capture (sec)")
    parser.add_argument("--pref_index", type=int, default=None, help="Preferred camera index to try first")
    parser.add_argument("--logical_res", default=f"{LOGICAL_RES[0]}x{LOGICAL_RES[1]}",
                        help="Logical resolution W×H, e.g., 1728x1117")
    parser.add_argument("--physical_res", default=f"{PHYSICAL_RES[0]}x{PHYSICAL_RES[1]}",
                        help="Physical resolution W×H, e.g., 3456x2234")
    args = parser.parse_args()

    # Parse resolutions
    def parse_res(s):
        m = re.match(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$", s)
        if not m:
            raise ValueError(f"Invalid resolution: {s}")
        return int(m.group(1)), int(m.group(2))

    logical_res  = parse_res(args.logical_res)
    physical_res = parse_res(args.physical_res)

    # Compute scale and convert logical pixel -> physical pixel
    sx, sy = compute_scale(logical_res, physical_res)
    px, py = logical_to_physical((args.x, args.y), (sx, sy))

    print(f"[INFO] Monitoring logical pixel {args.x, args.y} "
          f"-> physical pixel {px, py} with scale {(sx, sy)}")

    # Prepare output dir
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    # Open camera
    cap = open_camera_interactive(preferred_index=args.pref_index)

    # Setup MSS
    sct = mss.mss()

    # Trigger state
    ARMED, WAIT_RESET = 0, 1
    state = ARMED
    last_capture_ts = 0.0
    idx = get_next_index(out_dir, FNAME_PREFIX)

    # Preview window
    win = "Camera Preview (q=quit, c=calibrate, s=save)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    period = 1.0 / max(1e-6, args.poll)

    try:
        while True:
            t0 = time.time()

            # Grab camera frame
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[WARN] Failed to read camera frame; retrying...")
                time.sleep(0.05)
                continue

            # Read screen pixel
            rgb = sample_screen_pixel(sct, (px, py))

            # Draw small status HUD
            hud = f"Pixel RGB={rgb}   Target={TARGET_RGB}±{args.tol}   State={'ARMED' if state==ARMED else 'WAIT_RESET'}   NextIdx={idx}"
            cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if state==ARMED else (0,200,255), 2, cv2.LINE_AA)
            # Mark approximate pixel color box
            swatch = np.zeros((40, 60, 3), dtype=np.uint8)
            swatch[:] = (rgb[2], rgb[1], rgb[0])  # BGR for display
            frame[10:50, -70:-10] = cv2.resize(swatch, (60,40), interpolation=cv2.INTER_NEAREST)

            # Trigger logic
            if state == ARMED:
                if color_is_match(rgb, TARGET_RGB, args.tol):
                    # Save capture
                    fname = format_fname(FNAME_PREFIX, idx, FNAME_PAD_MIN)
                    fpath = os.path.join(out_dir, fname)
                    cv2.imwrite(fpath, frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                    print(f"[CAPTURE] Saved {fpath}")
                    idx += 1
                    last_capture_ts = time.time()
                    state = WAIT_RESET
            elif state == WAIT_RESET:
                # Debounce guard AND wait until pixel is not green
                if (time.time() - last_capture_ts) >= args.debounce and not color_is_match(rgb, TARGET_RGB, args.tol):
                    state = ARMED

            # Show preview
            cv2.imshow(win, frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('s'):
                # manual save
                fname = format_fname(FNAME_PREFIX, idx, FNAME_PAD_MIN)
                fpath = os.path.join(out_dir, fname)
                cv2.imwrite(fpath, frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                print(f"[MANUAL] Saved {fpath}")
                idx += 1
            elif k == ord('c'):
                sel = calibration_helper(sct, logical_res, physical_res)
                if sel is not None:
                    # Update logical and physical pixel
                    lx, ly = sel
                    px_new, py_new = logical_to_physical((lx, ly), (sx, sy))
                    print(f"[CALIB] Selected logical {sel} -> physical {(px_new, py_new)}")
                    px, py = px_new, py_new
                else:
                    print("[CALIB] Cancelled")

            # Maintain poll rate
            elapsed = time.time() - t0
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
