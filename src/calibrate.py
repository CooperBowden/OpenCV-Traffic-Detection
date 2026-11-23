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
    max_pts=10000,  # Master memory cap
):
    """Updates the master trajectory store using 2D spatial grid deduplication[cite: 2].

    Extracts the latest ground point from each sufficiently long active track,
    classifies movement direction along the vertical image axis, merges with
    existing historical points on disk, and caps spatial density so long-running
    collection does not exhaust memory[cite: 2].

    Args:
        track_history: Mapping of vehicle track IDs to ordered sequences of
            `(x, y)` ground-plane coordinates[cite: 2].
        store_file: Path to the `.npz` archive storing `"up"` and `"down"`
            point clouds. Defaults to `"trajectories_store.npz"`[cite: 2].
        grid_size: Side length in pixels of each square spatial deduplication
            bin. Defaults to 5[cite: 2].
        max_pts_per_cell: Maximum number of points retained per spatial grid
            cell. Defaults to 15[cite: 2].
        max_pts: Maximum total number of points retained per direction.
            Defaults to 10000[cite: 2].
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
    master_up = _deduplicate_spatially(
        merged_up, grid_size, max_pts_per_cell, max_pts
    )
    master_down = _deduplicate_spatially(
        merged_down, grid_size, max_pts_per_cell, max_pts
    )

    # 6. Save updated master lists back to disk
    np.savez(store_file, up=master_up, down=master_down)
    # print(f"[+] Master store updated: {len(master_up)} 'up' / {len(master_down)} 'down' total points.")


def _deduplicate_spatially(
    pts: np.ndarray, grid_size: int, max_per_cell: int, max_pts: int
) -> np.ndarray:
    """Bins 2D points into a spatial pixel grid and caps density per cell[cite: 2].

    Iterates in reverse chronological order so newer trajectory points are
    prioritized while preserving broad spatial road coverage within memory
    limits[cite: 2].

    Args:
        pts: Array of shape `(N, 2)` containing `(x, y)` coordinates ordered
            from oldest to newest[cite: 2].
        grid_size: Side length in pixels of each spatial grid cell[cite: 2].
        max_per_cell: Maximum number of points allowed in a single grid cell[cite: 2].
        max_pts: Maximum total number of points to retain across all cells[cite: 2].

    Returns:
        An array of shape `(M, 2)` containing the deduplicated points in
        chronological order, where `M <= max_pts`[cite: 2].
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
    """Vertically stacks historical and current 2D point arrays[cite: 2].

    Args:
        hist: Historical point array of shape `(N, 2)`[cite: 2].
        curr: Current frame point array of shape `(M, 2)`[cite: 2].

    Returns:
        A combined `(N + M, 2)` array, or whichever input array is non-empty[cite: 2].
    """
    if len(hist) > 0 and len(curr) > 0:
        return np.vstack((hist, curr))
    return curr if len(curr) > 0 else hist


def auto_calibrate(
    frame: np.ndarray,
    store_file="trajectories_store.npz",
    save_images: bool = False,
):
    """Automatically estimates upstream and downstream road bounds and lane counts[cite: 2].

    Loads historical vehicle trajectories, filters spatial outliers, fits
    perspective lane centerlines, and computes 4-point road bounding trapezoids
    for both traffic directions[cite: 2].

    Args:
        frame: BGR OpenCV video frame of the roadway[cite: 2].
        store_file: Path to the `.npz` file containing accumulated `"up"` and
            `"down"` point clouds. Defaults to `"trajectories_store.npz"`[cite: 2].
        save_images: Whether to save intermediate point cloud and lane peak
            visualizations to `../screenshots/`. Defaults to False[cite: 2].

    Returns:
        A 4-tuple `(coords_up, coords_down, num_lanes_up, num_lanes_down)`
        containing the upstream and downstream 4-point bounding arrays and
        their detected lane counts, or `(None, None)` if `store_file` does not
        exist[cite: 2].
    """
    basepath = os.path.dirname(__file__)
    if not os.path.exists(store_file):
        print(f"[!] Store file '{store_file}' not found.")
        return None, None

    data = np.load(store_file)
    up_pts = data.get("up", np.empty((0, 2), dtype=np.float32))
    down_pts = data.get("down", np.empty((0, 2), dtype=np.float32))

    filter_up = filter_pts(up_pts)
    filter_down = filter_pts(down_pts)
    point_cloud = plot_point_cloud_on_frame(
        frame=frame.copy(), radius=1, pts_up=filter_up, pts_down=filter_down
    )
    if save_images == True:
        cv2.imwrite(
            os.path.abspath(
                os.path.join(basepath, "..", "screenshots", "point_cloud.png")
            ),
            point_cloud,
        )
    print("filter up points: " + (str)(len(up_pts) - len(filter_up)))
    print("filter down points: " + (str)(len(down_pts) - len(filter_down)))
    annotated_up, lane_lines_up, num_lanes_up = plot_lane_peaks(
        frame=frame, pts=filter_up
    )
    annotated_down, lane_lines_down, num_lanes_down = plot_lane_peaks(
        frame=annotated_up, pts=filter_down
    )
    if save_images == True:
        cv2.imwrite(
            os.path.abspath(
                os.path.join(basepath, "..", "screenshots", "traffic_peaks.png")
            ),
            annotated_down,
        )
    coords_up = calculate_bounds(lane_lines_up)
    coords_down = calculate_bounds(lane_lines_down)
    return coords_up, coords_down, num_lanes_up, num_lanes_down


def detect_num_lanes(
    pts: np.ndarray,
    num_slices: int = 8,
    prominence_ratio: float = 0.05,
) -> int:
    """Estimates the number of active traffic lanes from a vehicle point cloud[cite: 2].

    Divides the vertical span of the point cloud into thin horizontal slices,
    corrects for overall road slope, and runs 1D smoothed histogram peak
    detection across each slice to determine the majority lane count without
    perspective slant smearing[cite: 2].

    Args:
        pts: Point cloud array of shape `(N, 2)` containing `(x, y)` vehicle
            positions[cite: 2].
        num_slices: Number of horizontal depth slices to evaluate. Defaults to 8[cite: 2].
        prominence_ratio: Minimum peak prominence expressed as a fraction of the
            maximum smoothed density in each slice. Defaults to 0.05[cite: 2].

    Returns:
        The estimated number of lanes (at least 1). Defaults to 3 if `pts`
        contains fewer than 15 points or yields no valid peaks[cite: 2].
    """
    if pts is None or len(pts) < 15:
        return 3

    # 1. Focus on the main active Y region (20th to 80th percentile)
    y_vals = pts[:, 1]
    y_min, y_max = np.percentile(y_vals, [20.0, 80.0])
    # account for slope of road
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

        # account for slope of road
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
    """Removes spatial and statistical outliers from a 2D vehicle point cloud[cite: 2].

    Applies a k-d tree radial neighbor density filter followed by axis-aligned
    interquartile range (IQR) filtering along both X and Y coordinates[cite: 2].

    Args:
        pts: Point cloud array of shape `(N, 2)` where column 0 is X and
            column 1 is Y[cite: 2].
        radius: Search radius in pixels for counting neighboring points.
            Defaults to 5[cite: 2].
        min_neighbors: Minimum number of neighbors within `radius` (excluding
            the point itself) required to keep a point. Defaults to 5[cite: 2].
        factor: IQR multiplier controlling outlier strictness (`1.0` for tight
            core traffic filtering, `1.5` for standard Tukey bounds, `2.0` to
            `2.5` for preserving wider lane changes). Defaults to 1.0[cite: 2].

    Returns:
        Filtered point cloud array of shape `(M, 2)` containing inlier points[cite: 2].
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
    """Fits and draws perspective lane centerlines from a vehicle point cloud[cite: 2].

    Extracts horizontal density peaks across depth slices and fits robust
    Huber regression lines for each detected lane so lines converge naturally
    toward the horizon vanishing point[cite: 2].

    Args:
        frame: BGR video frame onto which lane lines and labels are drawn[cite: 2].
        pts: Point cloud array of shape `(N, 2)` representing vehicle ground
            positions[cite: 2].
        num_slices: Number of horizontal depth slices used to sample lane
            centers. Defaults to 12[cite: 2].
        line_color: BGR color tuple for fitted lane centerlines. Defaults to
            `(0, 255, 255)`[cite: 2].
        peak_color: BGR color tuple for slice peak markers. Defaults to
            `(0, 0, 255)`[cite: 2].
        draw_peaks: Whether to draw circles at individual slice peak locations.
            Defaults to True[cite: 2].

    Returns:
        A 3-tuple `(annotated, lane_lines, num_lanes)` containing the annotated
        BGR frame, a list of `((x_top, y_top), (x_bottom, y_bottom))` line
        endpoint tuples, and the detected lane count[cite: 2]. Returns `(annotated, 0)`
        if `pts` has fewer than 20 points or an invalid vertical span[cite: 2].
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

        # estimate line position through avg x position in each slice
        line = cv2.fitLine(
            points=np.array(samples, dtype=np.float32),
            distType=cv2.DIST_HUBER,
            param=0,
            reps=0.01,
            aeps=0.01,
        )
        # extract to 1d list
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


def calculate_bounds(
    drawn_lines: list, padding_factor: float = 1.0
) -> np.ndarray:
    """Builds a 4-corner perspective road trapezoid from fitted lane centerlines[cite: 2].

    Expands outward from the outermost left and right lane centerlines by half
    an estimated lane width (scaled by `padding_factor`) at both the top and
    bottom vertical bounds[cite: 2].

    Args:
        drawn_lines: List of `(pt_top, pt_bottom)` line endpoint tuples[cite: 2].
        padding_factor: Outward width multiplier applied to the outer
            half-lane padding (`1.0` equals `0.5 * lane_width` per side).
            Defaults to 1.0[cite: 2].

    Returns:
        An integer array of shape `(4, 2)` ordered as `[[TL], [TR], [BR],
        [BL]]`, or an empty `(0, 2)` array if `drawn_lines` is empty[cite: 2].
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


def plot_point_cloud_on_frame(
    frame: np.ndarray,
    pts_up: np.ndarray = None,
    pts_down: np.ndarray = None,
    radius: int = 2,
):
    """Overlays upstream and downstream trajectory point clouds onto a frame[cite: 2].

    Draws upstream points in blue `(255, 100, 0)` and downstream points in
    orange `(0, 165, 255)` on a copy of the input BGR image[cite: 2].

    Args:
        frame: Input BGR OpenCV video frame[cite: 2].
        pts_up: Optional `(N, 2)` array of upstream `(x, y)` coordinates.
            Defaults to None[cite: 2].
        pts_down: Optional `(M, 2)` array of downstream `(x, y)` coordinates.
            Defaults to None[cite: 2].
        radius: Circle radius in pixels for each plotted point. Defaults to 2[cite: 2].

    Returns:
        A copy of `frame` with the trajectory points drawn onto it[cite: 2].
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