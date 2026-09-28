import numpy as np


def calculate_vehicle_speed(
    bev_history: list,
    bev_height: int,
    bev_width: int,
    num_dashes: int = 2,
    num_lanes: int = 1,
    fps: float = 30.0,
    margin_percent: float = 0.05,
) -> tuple[tuple[float, float], float, str]:
    """Calculates real vehicle speed (MPH), velocity vector, and cardinal heading

    using sub-frame virtual speed trap interpolation across user-drawn BEV bounds.

    Parameters:
    - bev_history: List/deque of tracked (x, y) coordinates in BEV space.
    - bev_height: Height of the BEV frame canvas (pixels).
    - bev_width: Width of the BEV frame canvas (pixels).
    - num_dashes: Number of dash-gap cycles spanning the drawn area (40 ft per cycle).
    - num_lanes: Number of lanes spanning the drawn area (12 ft per lane).
    - fps: Target video frame rate.
    - margin_percent: Inset safety margin from top/bottom drawn edges (default 0.05 = 5%).

    Returns:
    - (vx_mph, vy_mph): Velocity vector components in MPH.
    - speed_mph: Vehicle speed in MPH.
    - heading: Cardinal direction string (e.g., 'N', 'S', 'SE').
    """
    if len(bev_history) < 2 or fps <= 0 or bev_height <= 0:
        return (0.0, 0.0), 0.0

    pts = list(bev_history)

    # 1. Define Trap Line Y-coordinates (5% insets from drawn box boundaries)
    line_a_y = bev_height * margin_percent
    line_b_y = bev_height * (1.0 - margin_percent)

    # Real-world distance covered between the two trap lines (feet)
    total_feet = num_dashes * 40.0
    trap_distance_ft = total_feet * (1.0 - 2.0 * margin_percent)

    # 2. Scan track history for exact sub-frame crossing indices
    idx_a = None
    idx_b = None

    for i in range(1, len(pts)):
        y_prev, y_curr = pts[i - 1][1], pts[i][1]

        # Detect Line A crossing
        if (
            (y_prev < line_a_y <= y_curr) or (y_prev > line_a_y >= y_curr)
        ) and idx_a is None:
            dy = y_curr - y_prev
            ratio = (line_a_y - y_prev) / dy if dy != 0 else 0.0
            idx_a = (i - 1) + ratio

        # Detect Line B crossing
        if (
            (y_prev < line_b_y <= y_curr) or (y_prev > line_b_y >= y_curr)
        ) and idx_b is None:
            dy = y_curr - y_prev
            ratio = (line_b_y - y_prev) / dy if dy != 0 else 0.0
            idx_b = (i - 1) + ratio

    FTS_TO_MPH = 3600.0 / 5280.0  # Conversion constant (ft/s to mph)

    # 3. Calculate Speed
    if idx_a is not None and idx_b is not None:
        # Exact Speed Trap calculation (both lines crossed)
        frame_delta = abs(idx_b - idx_a)
        time_sec = frame_delta / fps
        speed_fts = trap_distance_ft / time_sec if time_sec > 0 else 0.0
        speed_mph = speed_fts * FTS_TO_MPH
    else:
        speed_mph = None
        vx_mph, vy_mph = None, None
    #     # Fallback: Instantaneous speed estimation while actively inside the box
    #     window = min(len(pts), 5)
    #     w_pts = pts[-window:]
    #     num_frames = len(w_pts) - 1
    #     time_sec = num_frames / fps
    #
    #     scale_y = total_feet / bev_height
    #     scale_x = (num_lanes * 12.0) / bev_width if bev_width > 0 else scale_y
    #
    #     dx_ft = (w_pts[-1][0] - w_pts[0][0]) * scale_x
    #     dy_ft = (w_pts[-1][1] - w_pts[0][1]) * scale_y
    #
    #     speed_fts = np.hypot(dx_ft, dy_ft) / time_sec if time_sec > 0 else 0.0
    #     speed_mph = speed_fts * FTS_TO_MPH
    if speed_mph != None:
    # Filter out tracking jitter for stationary objects
        if speed_mph < 1.0:
            return (0.0, 0.0), 0.0

        # 4. Compute Velocity Vector and
        dx_px = pts[-1][0] - pts[0][0]
        dy_px = pts[-1][1] - pts[0][1]

        total_px_dist = np.hypot(dx_px, dy_px)
        if total_px_dist > 0:
            vx_mph = (dx_px / total_px_dist) * speed_mph
            vy_mph = (dy_px / total_px_dist) * speed_mph
        else:
            vx_mph, vy_mph = 0.0, 0.0

    return (vx_mph, vy_mph), speed_mph