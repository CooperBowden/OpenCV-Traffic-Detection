import json
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from collections import Counter
import os
import numpy as np
import math


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
            min_speed = 30.0
            if dy > min_speed:
                # Y increases -> moving DOWN (towards camera)
                pts_down.append(latest_pt)
            elif dy < -min_speed:
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
    # print(f"[+] Master store updated: {len(master_up)} 'up' / {len(master_down)} 'down' total points.")


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


def auto_calibrate(
    frame: np.ndarray,
    store_file="trajectories_store.npz"
):
    if not os.path.exists(store_file):
        print(f"[!] Store file '{store_file}' not found.")
        return None, None

    data = np.load(store_file)
    up_pts = data.get("up", np.empty((0, 2), dtype=np.float32))
    down_pts = data.get("down", np.empty((0, 2), dtype=np.float32))

    filter_up = filter_pts(up_pts)
    filter_down = filter_pts(down_pts)
    point_cloud = plot_point_cloud_on_frame(frame=frame.copy(), radius=1, pts_up=filter_up, pts_down=filter_down)
    cv2.imshow("point cloud", point_cloud)
    print("filter up points: " + (str) (len(up_pts) - len(filter_up)))
    print("filter down points: " + (str) (len(down_pts) - len(filter_down)))
    lines_up, lane_lines_up, num_lanes_up  = plot_lane_peaks(frame=frame, pts=filter_up)
    lines_down, lane_lines_down, num_lanes_down = plot_lane_peaks(frame=lines_up, pts=filter_down)
    cv2.imshow("traffic peaks", lines_down)
    coords_up = calculate_bounds(lane_lines_up)
    coords_down = calculate_bounds(lane_lines_down)
    return coords_up, coords_down, num_lanes_up, num_lanes_down

def detect_num_lanes(
    pts: np.ndarray,
    num_slices: int = 8,
    prominence_ratio: float = 0.05,
) -> int:
    """Detects active lane count by running peak detection across multiple thin horizontal

    slices to eliminate perspective slant smearing.
    """
    if pts is None or len(pts) < 15:
        return 3

    # 1. Focus on the main active Y region (20th to 80th percentile)
    y_vals = pts[:, 1]
    y_min, y_max = np.percentile(y_vals, [20.0, 80.0])
    #account for slope of road
    m_slope, _ = np.polyfit(pts[:, 1], pts[:, 0], deg=1)
    slice_edges = np.linspace(y_min, y_max, num_slices + 1)
    detected_counts = []

    # 2. Process each THIN horizontal slice independently
    for i in range(num_slices):
        y_start, y_end = slice_edges[i], slice_edges[i + 1]
        y_center = (y_start + y_end) / 2.0
        slice_mask = (y_vals >= y_start) & (y_vals < y_end)
        slice_pts = pts[slice_mask]

        if len(slice_pts) < 10:
            continue

        #account for slope of road
        x_coords = slice_pts[:, 0] - m_slope * (slice_pts[:, 1] - y_center)
        # Filter extreme horizontal outliers in this slice
        p2, p98 = np.percentile(x_coords, [2.0, 98.0])
        valid_x = x_coords[(x_coords >= p2) & (x_coords <= p98)]

        if len(valid_x) < 8:
            continue

        x_min, x_max = np.min(valid_x), np.max(valid_x)
        span = int(x_max - x_min)

        if span <= 15:
            continue

        # 1D histogram within thin slice (minimal Y variation = zero slant blur)
        num_bins = min(span, 300)
        hist, bin_edges = np.histogram(
            valid_x, bins=num_bins, range=(x_min, x_max)
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        # Smooth histogram
        smoothed = gaussian_filter1d(hist.astype(np.float32), 1.0)
        bin_width = (x_max - x_min) / num_bins
        distance = max(1, int(40 / bin_width))
        # Detect peaks with lowered prominence (0.05) to capture sparser lanes
        max_density = np.max(smoothed)
        peaks, _ = find_peaks(
            smoothed,
            distance=distance,
            prominence=max_density * prominence_ratio,
        )

        if len(peaks) > 0:
            detected_counts.append(len(peaks))

    if not detected_counts:
        return 3

    # Find majority number of lanes per slice and take that
    avg_lanes = round(sum(detected_counts) / len(detected_counts))
    counts_counter = Counter(detected_counts)
    majority_lanes = counts_counter.most_common(1)[0][0]
    # Find max lines detected
    # lanes_count = max(detected_counts)
    print(
        f"num lanes detected per slice: {detected_counts} -> Selected:"
        f" {avg_lanes} lanes"
    )

    return max(majority_lanes, 1)


def filter_pts(pts, radius=5, min_neighbors=5, factor: float = 1.0):
    """
    pts (np.ndarray): Point cloud array of shape (N, 2) where column 0 is X and column 1 is Y.
    radius (int): Radius of neighboring points.
    min_neighbors (int): Minimum number of neighboring points.
    factor (float): IQR multiplier:
                    - 1.5: Standard Tukey threshold (strips moderate & extreme outliers).
                    - 2.0 - 2.5: Milder filter (preserves wider lane-changes).
                    - 1.0: Aggressive filter (tightly keeps core traffic).
    :return: np.ndarray
    """
    tree = cKDTree(pts)
    # Count neighbors within radius (excluding self)
    neighbor_counts = tree.query_ball_point(pts, r=radius, return_length=True)
    mask = (neighbor_counts - 1) >= min_neighbors
    pts = pts[mask]
    mask = np.ones(len(pts), dtype=bool)

    # 1. Filter along X-axis (Horizontal off-road / shoulder outliers)
    q1_x, q3_x = np.percentile(pts[:, 0], [25, 75])
    iqr_x = q3_x - q1_x
    lower_x = q1_x - factor * iqr_x
    upper_x = q3_x + factor * iqr_x
    mask &= (pts[:, 0] >= lower_x) & (pts[:, 0] <= upper_x)

    # 2. Filter along Y-axis (Vertical far-away / horizon outliers)
    q1_y, q3_y = np.percentile(pts[:, 1], [25, 75])
    iqr_y = q3_y - q1_y
    lower_y = q1_y - factor * iqr_y
    upper_y = q3_y + factor * iqr_y
    mask &= (pts[:, 1] >= lower_y) & (pts[:, 1] <= upper_y)

    return pts[mask]

def plot_lane_peaks(
    frame: np.ndarray,
    pts: np.ndarray,
    num_slices: int = 12,
    line_color: tuple = (0, 255, 255),  # Yellow (BGR)
    peak_color: tuple = (0, 0, 255),  # Red (BGR)
    draw_peaks: bool = True,
) -> tuple[np.ndarray, int]:
    """Fits perspective centerlines that converge to a shared horizon Vanishing Point

        so lines naturally get closer together near the top of the road.
        """
    annotated = frame.copy()

    if pts is None or len(pts) < 20:
        return annotated, 0

    # 1. Establish perspective Y-range (Top to Bottom)
    y_min = float(np.percentile(pts[:, 1], 10))  # Top of road (near horizon)
    y_max = float(np.percentile(pts[:, 1], 90))  # Bottom of road (foreground)

    if y_max <= y_min:
        return annotated, 0

    # Auto-detect lane count
    num_lanes = detect_num_lanes(pts)

    # 2. Extract slice peaks (red dots) for each lane along Y
    slice_edges = np.linspace(y_min, y_max, num_slices + 1)
    lane_samples = [[] for _ in range(num_lanes)]  # lane_samples[i] holds red dots for Lane i

    for k in range(num_slices):
        y_start, y_end = slice_edges[k], slice_edges[k + 1]
        y_center = (y_start + y_end) / 2.0

        slice_pts = pts[(pts[:, 1] >= y_start) & (pts[:, 1] < y_end)]
        if len(slice_pts) < 5:
            continue

        x_coords = slice_pts[:, 0]
        p2, p98 = np.percentile(x_coords, [2.0, 98.0])
        valid_x = x_coords[(x_coords >= p2) & (x_coords <= p98)]

        if len(valid_x) < 5:
            continue

        x_min, x_max = np.min(valid_x), np.max(valid_x)
        span = int(x_max - x_min)
        if span <= 5:
            continue

        num_bins = min(span, 250)
        hist, bin_edges = np.histogram(
            valid_x, bins=num_bins, range=(x_min, x_max)
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        smoothed = gaussian_filter1d(hist.astype(np.float32), sigma=3.0)

        # Dynamic peak distance scaling by depth
        y_depth_ratio = (y_center - y_min) / max(y_max - y_min, 1.0)
        min_peak_dist = max(
            (num_bins / (num_lanes * 2.5)) * (0.3 + 0.7 * y_depth_ratio), 3.0
        )

        peaks, props = find_peaks(
            smoothed,
            distance=min_peak_dist,
            prominence=np.max(smoothed) * 0.05,
        )

        if len(peaks) >= num_lanes:
            top_idxs = np.argsort(props["prominences"])[-num_lanes:]
            extracted_peaks = sorted(bin_centers[peaks[top_idxs]])
        else:
            p_low, p_high = np.percentile(valid_x, [10, 90])
            extracted_peaks = np.linspace(p_low, p_high, num_lanes)

        # Assign peaks from left to right: Lane 0 (left), Lane 1 (mid), Lane 2 (right)
        for lane_idx in range(num_lanes):
            lane_samples[lane_idx].append(
                (float(extracted_peaks[lane_idx]), float(y_center))
            )
    lane_lines = []
    # 3. Fit Linear Regression (X = m*Y + b) per lane through its red dots
    for lane_idx, samples in enumerate(lane_samples):
        if len(samples) < 2:
            continue

        samples_arr = np.array(samples)
        x_s = samples_arr[:, 0]  # X coordinates of red dots for this lane
        y_s = samples_arr[:, 1]  # Y coordinates of red dots for this lane

        #estimate line position through avg x position in each slice
        line = cv2.fitLine(
            points=np.array(samples, dtype=np.float32),
            distType=cv2.DIST_HUBER,
            param=0,
            reps=0.01,
            aeps=0.01,
        )
        #extract to 1d list
        vx, vy, x0, y0 = line.ravel().tolist()

        # Line equation: X = x0 + (vx/vy)*(Y - y0)
        slope_x = vx / vy
        x_top = int(x0 + slope_x * (y_min - y0))
        x_bottom = int(x0 + slope_x * (y_max - y0))

        pt_top = (x_top, int(y_min))
        pt_bottom = (x_bottom, int(y_max))
        lane_lines.append((pt_top, pt_bottom))

        # Draw average line down the center of this lane's red dots
        cv2.line(
            annotated,
            pt_top,
            pt_bottom,
            color=line_color,
            thickness=3,
            lineType=cv2.LINE_AA,
        )

        # Optionally draw the red slice peak dots
        if draw_peaks:
            for sx, sy in samples:
                cv2.circle(
                    annotated,
                    (int(sx), int(sy)),
                    radius=4,
                    color=peak_color,
                    thickness=-1,
                )

        # Label each lane at the bottom
        cv2.putText(
            annotated,
            f"Lane {lane_idx + 1}",
            (pt_bottom[0] - 25, pt_bottom[1] + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return annotated, lane_lines, num_lanes

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


def calculate_bounds(
    drawn_lines: list, padding_factor: float = 1.0
) -> np.ndarray:
    """Creates a 4-corner perspective road bounding box directly from drawn line endpoints.

    Parameters:
        drawn_lines (list): List of line tuples [((x_top1, y_top1), (x_bot1,
          y_bot1)), ...]
        padding_factor (float): Outward padding multiplier (1.0 = 0.5 * lane_width per side).

    Returns:
        np.ndarray: 4-point trapezoid array [[TL], [TR], [BR], [BL]]
    """
    if not drawn_lines:
        return np.empty((0, 2), dtype=np.int32)
    # 1. Collect top and bottom X-coordinates across all drawn lines
    tops = []
    bottoms = []

    for pt_top, pt_bottom in drawn_lines:
        tops.append((pt_top[0], pt_top[1]))
        bottoms.append((pt_bottom[0], pt_bottom[1]))

    # Sort lines from left to right based on bottom X-coordinate
    tops.sort(key=lambda p: p[0])
    bottoms.sort(key=lambda p: p[0])

    y_min = tops[0][1]  # Horizon Y
    y_max = bottoms[0][1]  # Foreground Y

    num_lanes = len(drawn_lines)

    # 2. Extract furthest-left and furthest-right centerline X positions
    x_left_top = tops[0][0]
    x_right_top = tops[-1][0]

    x_left_bot = bottoms[0][0]
    x_right_bot = bottoms[-1][0]

    # 3. Calculate average lane spacing at top and bottom
    if num_lanes > 1:
        w_top = (x_right_top - x_left_top) / (num_lanes - 1)
        w_bot = (x_right_bot - x_left_bot) / (num_lanes - 1)
    else:
        w_top, w_bot = 80.0, 200.0  # Default fallbacks for single-lane

    # 4. Compute outer half-lane padding (0.5 * lane_width)
    half_top = (w_top / 2.0) * padding_factor
    half_bot = (w_bot / 2.0) * padding_factor

    # 5. Build 4-corner perspective road bounding box
    bbox = np.array(
        [
            [int(x_left_top - half_top), int(y_min)],  # Top-Left (TL)
            [int(x_right_top + half_top), int(y_min)],  # Top-Right (TR)
            [int(x_right_bot + half_bot), int(y_max)],  # Bottom-Right (BR)
            [int(x_left_bot - half_bot), int(y_max)],  # Bottom-Left (BL)
        ],
        dtype=np.int32,
    )

    return bbox

def get_length(
        bev_frame: np.ndarray, display_scale, num_lanes: int,
) -> float:
    """Accurately counts painted dashes in BEV space by sampling narrow vertical
    strips along lane divider lines rather than averaging the full image width.
    Uses masked BEV frame to remove vehicles to isolate dashes
    """
    if bev_frame is None or bev_frame.size == 0:
        return 3.0

    if len(bev_frame.shape) == 3:
        gray = cv2.cvtColor(bev_frame, cv2.COLOR_BGR2GRAY)
        viz_frame = bev_frame.copy()
    else:
        gray = bev_frame.copy()
        viz_frame = cv2.cvtColor(bev_frame, cv2.COLOR_GRAY2BGR)

    h, w = gray.shape[:2]

    # 1. Thresholding to isolate bright white road paint
    _, binary = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)

    # Vertical opening to eliminate horizontal noise/reflections
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 2. Locate isolated lane divider X positions
    lane_width_px = w / float(num_lanes)
    divider_x_positions = [int(i * lane_width_px) for i in range(1, num_lanes)]
    if not divider_x_positions:
        divider_x_positions = [int(w / 2.0)]

    strip_w = 30  # Half-width of vertical strip around divider line
    dash_lengths = []

    for div_x in divider_x_positions:
        x_start = max(0, div_x - strip_w)
        x_end = min(w, div_x + strip_w)

        # Slice narrow isolated vertical strip along the divider line
        strip = binary[:, x_start:x_end]

        # Extract contours within this specific lane segment
        contours, _ = cv2.findContours(
            strip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Filter for vertically elongated dashes (height >= 10px, height > width)
            if bh >= 10 and bh > bw and bw <= 35:
                # Map local strip coordinates back to global BEV frame coordinates
                x_mid_global = x_start + x + (bw // 2)
                pt_start = (x_mid_global, y)
                pt_end = (x_mid_global, y + bh)
                # find distance of points and add to list
                dash_lengths.append((math.dist(pt_start, pt_end)/ display_scale))
                # Draw single centerline per dash
                cv2.line(
                    viz_frame,
                    pt_start,
                    pt_end,
                    (0, 255, 0),
                    2,
                    lineType=cv2.LINE_AA,
                )
    # return 0 if can't calculate
    if (len(dash_lengths) == 0):
        return 0

    # filter outliers/clipped dashes
    def remove_outliers_iqr(list):
        q1 = np.percentile(list, 25)
        q3 = np.percentile(list, 75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Filter the list
        return [x for x in list if lower_bound <= x <= upper_bound]
    filtered_lengths = remove_outliers_iqr(dash_lengths)
    avg_length = sum(filtered_lengths) / float(len(filtered_lengths))

    ft_per_px = 10.0 / avg_length  # how many feet per pixel of the frame
    frame_length_px = bev_frame.shape[:2][0]/display_scale
    frame_length_ft = frame_length_px * ft_per_px  # length of frame in feet

    cv2.imshow("frame with lanes", viz_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print(f"Estimated length: {frame_length_ft:.2f}ft")
    return frame_length_ft

def plot_point_cloud_on_frame(
    frame: np.ndarray,
    pts_up: np.ndarray = None,
    pts_down: np.ndarray = None,
    radius: int = 2,
):
    """Overlays directional point clouds onto an OpenCV video frame.

    - Upstream traffic (pts_up): Blue
    - Downstream traffic (pts_down): Orange
    """
    canvas = frame.copy()

    # Draw 'up' trajectory points (Blue)
    if pts_up is not None and len(pts_up) > 0:
        for x, y in pts_up.astype(int):
            cv2.circle(canvas, (x, y), radius, (255, 100, 0), -1)

    # Draw 'down' trajectory points (Orange)
    if pts_down is not None and len(pts_down) > 0:
        for x, y in pts_down.astype(int):
            cv2.circle(canvas, (x, y), radius, (0, 165, 255), -1)

    return canvas