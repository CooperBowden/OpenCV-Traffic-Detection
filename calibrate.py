import json
import cv2
import os
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


import os
import numpy as np

import os
import numpy as np


def update_master_trajectory_store(
        track_history,
        store_file="trajectories_store.npz",
        grid_size=5,  # Bin points into 5x5 pixel spatial cells
        max_pts_per_cell=15,  # Cap points per cell so lanes don't over-fill
        max_pts=10000  # Master memory cap
):
    """
    Updates the master trajectory store without duplicate point dumping.
    Uses 2D Spatial Grid Deduplication to allow data collection over hours/days.
    """
    pts_up = []
    pts_down = []

    # 1. Collect ONLY the latest single ground point (pts[-1]) per active track
    for track_id, pts in track_history.items():
        if len(pts) >= 5:  # Require minimum tracking length
            x_start, y_start = pts[0]
            x_end, y_end = pts[-1]

            dy = y_end - y_start  # Vertical movement delta

            # Extract ONLY the newest point (pts[-1]) to prevent quadratic point duplication
            latest_pt = pts[-1]

            if dy > 2.0:
                # Y increases -> moving DOWN (towards camera)
                pts_down.append(latest_pt)
            elif dy < -2.0:
                # Y decreases -> moving UP (towards horizon)
                pts_up.append(latest_pt)

    # 2. Convert frame batch into 2D float NumPy arrays
    curr_up = (
        np.array(pts_up, dtype=np.float32).reshape(-1, 2)
        if len(pts_up) > 0
        else np.empty((0, 2), dtype=np.float32)
    )
    curr_down = (
        np.array(pts_down, dtype=np.float32).reshape(-1, 2)
        if len(pts_down) > 0
        else np.empty((0, 2), dtype=np.float32)
    )

    # 3. Load pre-existing historical master points
    hist_up = np.empty((0, 2), dtype=np.float32)
    hist_down = np.empty((0, 2), dtype=np.float32)

    if os.path.exists(store_file):
        try:
            data = np.load(store_file)
            hist_up = data.get("up", hist_up)
            hist_down = data.get("down", hist_down)
        except Exception as e:
            print(f"[!] Warning: Could not read '{store_file}': {e}")

    # 4. Merge current frame points with historical store
    merged_up = _combine_arrays(hist_up, curr_up)
    merged_down = _combine_arrays(hist_down, curr_down)

    # 5. Apply SPATIAL GRID DEDUPLICATION (Keeps spatial coverage, prevents filling up)
    master_up = _deduplicate_spatially(merged_up, grid_size, max_pts_per_cell, max_pts)
    master_down = _deduplicate_spatially(merged_down, grid_size, max_pts_per_cell, max_pts)

    # 6. Save updated master lists back to disk
    np.savez(store_file, up=master_up, down=master_down)
    print(f"[+] Master store updated: {len(master_up)} 'up' / {len(master_down)} 'down' total points.")


def _deduplicate_spatially(pts: np.ndarray, grid_size: int, max_per_cell: int, max_pts: int) -> np.ndarray:
    """
    Bins 2D points into a spatial pixel grid (e.g. 5x5 px).
    Limits point count per grid cell so long runs collect wide road geometry
    without filling memory buffers on repetitive traffic paths.
    """
    if len(pts) == 0:
        return pts

    # Quantize coordinates into integer spatial grid cells
    grid_coords = (pts / grid_size).astype(int)

    cell_counts = {}
    kept_indices = []

    # Iterate in reverse (newest points first) to prioritize recent tracks
    for idx in range(len(pts) - 1, -1, -1):
        cell_key = (grid_coords[idx, 0], grid_coords[idx, 1])
        count = cell_counts.get(cell_key, 0)

        if count < max_per_cell:
            cell_counts[cell_key] = count + 1
            kept_indices.append(idx)

        if len(kept_indices) >= max_pts:
            break

    # Restore chronological order
    kept_indices.reverse()
    return pts[kept_indices]


def _combine_arrays(hist: np.ndarray, curr: np.ndarray) -> np.ndarray:
    """Helper to safely stack historical and current point matrices."""
    if len(hist) > 0 and len(curr) > 0:
        return np.vstack((hist, curr))
    return curr if len(curr) > 0 else hist

def calibrate(
    frame: np.ndarray,
    camera: str,
    scale: float,
):
    def trackbar_event(val):
        pass

    def redraw():
        """Redraws all points and connecting lines on a fresh frame copy."""
        nonlocal frame_display
        frame_display = frame.copy()

        num_pts = len(clicked_points)
        if num_pts == 0:
            return

        # 1. Draw lines connecting consecutive points
        for i in range(num_pts - 1):
            pt1 = tuple((clicked_points[i] * scale).astype(int))
            pt2 = tuple((clicked_points[i + 1] * scale).astype(int))
            cv2.line(frame_display, pt1, pt2, (255, 0, 0), 1, cv2.LINE_AA)

        # 2. Close the box if all 4 points are set
        if num_pts == 4:
            pt_first = tuple((clicked_points[0] * scale).astype(int))
            pt_last = tuple((clicked_points[3] * scale).astype(int))
            cv2.line(
                frame_display, pt_last, pt_first, (255, 0, 0), 1, cv2.LINE_AA
            )

        # 3. Draw circles and text labels for every point
        for i, pt in enumerate(clicked_points):
            px, py = int(pt[0] * scale), int(pt[1] * scale)
            # cv2.circle(frame_display, (px, py), 5, (0, 0, 255), -1)
            cv2.drawMarker(frame_display, (px, py), (0, 0, 255), cv2.LINE_AA)
            # cv2.putText(
            #     frame_display,
            #     f"P{i + 1}: ({px}, {py})",
            #     (px + 10, py - 10),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5,
            #     (0, 255, 0),
            #     2,
            # )

    def click_event(event, x, y, flags, param):
        """Callback function to handle mouse clicks on the video frame."""
        nonlocal clicked_points
        if event == cv2.EVENT_LBUTTONDOWN:
            new_point = np.array([int(x / scale), int(y / scale)])

            if len(clicked_points) == 4:
                # Compute Euclidean distances and replace closest point
                distances_sq = np.sum((clicked_points - new_point) ** 2, axis=1)
                closest_idx = np.argmin(distances_sq)
                clicked_points[closest_idx] = new_point
            else:
                # Append new point
                if len(clicked_points) == 0:
                    clicked_points = new_point.reshape(1, 2)
                else:
                    clicked_points = np.vstack([clicked_points, new_point])

            redraw()

    try:
        with open("config.json", "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {}

    if camera not in data:
        data[camera] = {}

    # Initialize clicked_points array
    clicked_points = np.empty((0, 2), dtype=int)

    # Base frame copy for drawing
    frame_display = frame.copy()

    # Setup the OpenCV Window and Mouse Callback
    cv2.namedWindow("Select 4 Ground Points")
    cv2.setMouseCallback("Select 4 Ground Points", click_event)
    try:
        with open("config.json", "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {}
    # Trackbars
    cv2.createTrackbar(
        "Number of lanes", "Select 4 Ground Points", data[camera]["num_lanes"], 5, trackbar_event
    )
    cv2.createTrackbar(
        "Number of dashes", "Select 4 Ground Points", data[camera]["num_lanes"], 10, trackbar_event
    )

    print("INSTRUCTIONS:")
    print("Click 4 points on the ground plane in CLOCKWISE order:")
    print(
        "1. Top-Left  ->  2. Top-Right  ->  3. Bottom-Right  ->  4. Bottom-Left"
    )
    print("Clicking a 5th time replaces the closest point.")
    print("Press 'r' to reset points. Press 's' to save. Press 'q' to quit.")

    # Calibration loop
    while True:
        display_frame = frame_display.copy()

        num_lanes = cv2.getTrackbarPos(
            "Number of lanes", "Select 4 Ground Points"
        )
        num_dashes = cv2.getTrackbarPos(
            "Number of dashes", "Select 4 Ground Points"
        )

        # Draw trackbar info overlay (split across lines since cv2.putText doesn't handle \n)
        cv2.putText(
            display_frame,
            f"Number of lanes: {num_lanes}",
            (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            display_frame,
            f"Number of dashes: {num_dashes}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Select 4 Ground Points", display_frame)
        key = cv2.waitKey(30) & 0xFF

        # Reset choices
        if key == ord("r"):
            clicked_points = np.empty((0, 2), dtype=int)
            redraw()
            print("Points reset. Start over from Top-Left.")

        # Quit without saving
        if key == ord("q"):
            cv2.destroyWindow("Select 4 Ground Points")
            return None

        # Save configuration
        if key == ord("s"):
            if len(clicked_points) < 4:
                print("Please select 4 points before saving.")
                continue

            print("Points saved:")
            print(f"src_pts = np.float32(\n{np.float32(clicked_points)}\n)")

            data[camera]["points"] = clicked_points.tolist()
            data[camera]["num_lanes"] = num_lanes
            data[camera]["num_dashes"] = num_dashes

            with open("config.json", "w") as file:
                json.dump(data, file, indent=4)

            cv2.destroyWindow("Select 4 Ground Points")
            return clicked_points, num_lanes, num_dashes



def auto_calibrate_from_trajectories(
        store_file="trajectories_store.npz"
):
    if not os.path.exists(store_file):
        print(f"[!] Store file '{store_file}' not found.")
        return None, None

    data = np.load(store_file)
    master_up = data.get("up", np.empty((0, 2), dtype=np.float32))
    master_down = data.get("down", np.empty((0, 2), dtype=np.float32))

    coords_up = calculate_bounds(master_up) if len(master_up) > 0 else None
    coords_down = calculate_bounds(master_down) if len(master_down) > 0 else None

    return coords_up, coords_down



def extract_lane_boundaries_from_band(
    band_pts, num_lanes=3, x_min_clip=0, x_max_clip=1920
):
    """Extracts mathematically stable lane boundary X-coordinates (valleys & outer edges)

    from a 2D point cloud band without width inflation or outlier distortions.
    """
    if band_pts is None or len(band_pts) == 0:
        return None

    # 1. Filter extreme horizontal outliers (e.g., tracking errors on shoulders)
    x_raw = band_pts[:, 0]
    p_low, p_high = np.percentile(x_raw, [1.0, 99.0])
    valid_mask = (x_raw >= p_low) & (x_raw <= p_high)
    x_coords = x_raw[valid_mask]

    if len(x_coords) < 10:
        x_coords = x_raw

    # 2. Build 1D histogram over the filtered horizontal span
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    span = int(x_max - x_min)
    if span <= 10:
        return None

    num_bins = min(span, 500)
    hist, bin_edges = np.histogram(x_coords, bins=num_bins, range=(x_min, x_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Smooth histogram with Gaussian filter
    smoothed = gaussian_filter1d(hist.astype(np.float32), sigma=5.0)

    # 3. Detect Peaks (Lane Centers)
    min_dist = max(num_bins / (num_lanes * 2.5), 5.0)
    peak_indices, properties = find_peaks(
        smoothed, distance=min_dist, prominence=np.max(smoothed) * 0.05
    )

    if len(peak_indices) >= num_lanes:
        # Select the 'num_lanes' MOST PROMINENT peaks (filters out shoulder noise)
        prominences = properties["prominences"]
        top_indices = np.argsort(prominences)[-num_lanes:]
        peaks = sorted(bin_centers[peak_indices[top_indices]])
    else:
        # Fallback: Estimate lane centers from percentile span
        p_first = np.percentile(x_coords, 5)
        p_last = np.percentile(x_coords, 95)
        peaks = np.linspace(p_first, p_last, num_lanes)

    # 4. Calculate robust average pixel width per lane
    if num_lanes > 1:
        avg_lane_w = (peaks[-1] - peaks[0]) / (num_lanes - 1)
    else:
        avg_lane_w = span

    # 5. Locate internal valleys (lane dividers) or fall back to midpoints
    valleys = []
    for i in range(num_lanes - 1):
        p1, p2 = peaks[i], peaks[i + 1]
        mask = (bin_centers >= p1) & (bin_centers <= p2)
        if np.any(mask):
            segment = smoothed[mask]
            seg_centers = bin_centers[mask]
            valleys.append(seg_centers[np.argmin(segment)])
        else:
            valleys.append((p1 + p2) / 2.0)

    # 6. Set outer boundaries using exactly 0.5 * avg_lane_w outward push
    b_left = peaks[0] - 0.5 * avg_lane_w
    b_right = peaks[-1] + 0.5 * avg_lane_w

    # Assemble ordered boundary vector [B0, V1, V2, ..., BN]
    boundaries = [int(b_left)] + [int(v) for v in valleys] + [int(b_right)]

    # Clip boundaries to image dimensions
    boundaries = [
        int(max(x_min_clip, min(x_max_clip, b))) for b in boundaries
    ]
    return boundaries


def calculate_bounds(pts, num_lanes=3):
    """Constructs road trapezoid bounds using outlier-filtered band analysis."""
    if pts is None or len(pts) == 0:
        return None

    # 1. Establish Y bands (Horizon vs Camera foreground)
    y_min = np.percentile(pts[:, 1], 35)
    y_max = np.percentile(pts[:, 1], 95)

    band_h = max((y_max - y_min) * 0.15, 10.0)
    top_band = pts[np.abs(pts[:, 1] - y_min) < band_h]
    bottom_band = pts[np.abs(pts[:, 1] - y_max) < band_h]

    if len(top_band) == 0 or len(bottom_band) == 0:
        return None

    # 2. Extract X boundaries at top and bottom heights
    top_bounds = extract_lane_boundaries_from_band(top_band, num_lanes=num_lanes)
    bottom_bounds = extract_lane_boundaries_from_band(
        bottom_band, num_lanes=num_lanes
    )

    if top_bounds is None or bottom_bounds is None:
        return None

    # 3. Assemble overall road trapezoid [TL, TR, BR, BL]
    master_road_coords = np.array(
        [
            [top_bounds[0], int(y_min)],  # Top-Left
            [top_bounds[-1], int(y_min)],  # Top-Right
            [bottom_bounds[-1], int(y_max)],  # Bottom-Right
            [bottom_bounds[0], int(y_max)],  # Bottom-Left
        ],
        dtype=np.int32,
    )

    return master_road_coords