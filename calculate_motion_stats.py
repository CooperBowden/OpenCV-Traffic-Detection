import numpy as np


def calculate_vehicle_speed(
    bev_history: list,
    bev_height: int,
    length: int,
    width: int,
    direction: str,
    fps: float = 30.0,
    margin_percent: float = 0.05,
) -> tuple[tuple[float, float], float]:
    """Calculates real vehicle speed (MPH), velocity vector, and cardinal heading

    using sub-frame virtual speed trap interpolation across user-drawn BEV bounds.

    Parameters:
    - bev_history: List/deque of tracked (x, y) coordinates in BEV space.
    - bev_height: Height of the BEV frame canvas (pixels).
    - length: box length in feet
    - fps: Target video frame rate.
    - margin_percent: Inset safety margin from top/bottom drawn edges (default 0.05 = 5%).

    Returns:
    - (vx_mph, vy_mph): Velocity vector components in MPH.
    - speed_mph: Vehicle speed in MPH.
    """
    if len(bev_history) < 2 or fps <= 0 or bev_height <= 0:
        return (0.0, 0.0), None

    pts = list(bev_history)
    if (direction == "down"):
        # 1. Define Trap Line Y-coordinates (5% insets from drawn box boundaries)
        line_a_y = bev_height * margin_percent
        line_b_y = bev_height * (1.0 - margin_percent)
    else:
        # 1. Define Trap Line Y-coordinates (5% insets from drawn box boundaries)
        line_a_y = bev_height * (1.0 - margin_percent)
        line_b_y = bev_height * margin_percent

    # Real-world distance covered between the two trap lines (feet)
    trap_distance_ft = length * (1.0 - 2.0 * margin_percent)

    # 2. Scan track history for exact sub-frame crossing indices
    y_prev_step = pts[-2][1]
    y_curr_step = pts[-1][1]

    just_crossed_b = (y_prev_step < line_b_y <= y_curr_step) or (
            y_prev_step > line_b_y >= y_curr_step
    )

    # If Line B was not crossed on this frame step, exit early
    if not just_crossed_b:
        return (0.0, 0.0), None

    idx_a = None
    # fractional frame where line b was crossed
    idx_b = (len(pts) - 2) + (
        (line_b_y - y_prev_step) / (y_curr_step - y_prev_step)
        if (y_curr_step - y_prev_step) != 0
        else 0.0
    )
    for i in range(1, len(pts)):
        y_prev, y_curr = pts[i - 1][1], pts[i][1]
        if (
                (y_prev < line_a_y <= y_curr) or (y_prev > line_a_y >= y_curr)
        ) and idx_a is None:
            dy = y_curr - y_prev
            ratio = (line_a_y - y_prev) / dy if dy != 0 else 0.0
            idx_a = (i - 1) + ratio
            break
    if idx_a is None:
        return (0.0, 0.0), None

    # 3. Calculate Speed
    FTS_TO_MPH = 3600.0 / 5280.0  # Conversion constant (ft/s to mph)

    if idx_a is not None and idx_b is not None:
        # Exact Speed Trap calculation (both lines crossed)
        frame_delta = abs(idx_b - idx_a)
        time_sec = frame_delta / fps
        speed_fts = trap_distance_ft / time_sec if time_sec > 0 else 0.0
        speed_mph = speed_fts * FTS_TO_MPH
    else:
        speed_mph = None
        vx_mph, vy_mph = None, None
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

def calculate_real_time_speed(
    history: list,
    scale_x: float,  # feet per pixel (X axis)
    scale_y: float,  # feet per pixel (Y axis)
    direction: str,
    fps: float,
    window: int = 5,
) -> tuple[tuple[float, float], float, str]:
    """Computes:

    - (vx_mph, vy_mph): velocity vector in MPH
    - speed_mph: magnitude in MPH
    - heading: cardinal direction
    """
    if len(history) < 2 or fps <= 0:
        return (0.0, 0.0), 0.0, "--"

    pts = list(history)
    w_pts = pts[-min(len(pts), window) :]
    num_frames = len(w_pts) - 1

    if num_frames == 0:
        return (0.0, 0.0), 0.0, "--"

    # Time elapsed in seconds across sliding window
    time_sec = num_frames / fps

    # Displacement in real-world feet
    dx_ft = (w_pts[-1][0] - w_pts[0][0]) * scale_x
    dy_ft = (w_pts[-1][1] - w_pts[0][1]) * scale_y

    # Velocity in feet per second (ft/s)
    vx_fts = dx_ft / time_sec
    vy_fts = dy_ft / time_sec
    speed_fts = np.hypot(vx_fts, vy_fts)

    # Conversion factor: 1 ft/s = (3600 / 5280) mph = 0.681818 mph
    FTS_TO_MPH = 3600.0 / 5280.0
    speed_mph = float(speed_fts * FTS_TO_MPH)
    vx_mph = vx_fts * FTS_TO_MPH
    vy_mph = vy_fts * FTS_TO_MPH

    # Stationary threshold (~1.5 MPH to filter tracking jitter)
    if speed_mph < 1.5:
        return (0.0, 0.0), 0.0, "Stopped"

    # Cardinal heading angle (using pixel displacement)
    dx_px = w_pts[-1][0] - w_pts[0][0]
    dy_px = w_pts[-1][1] - w_pts[0][1]
    angle = np.degrees(np.arctan2(dy_px, dx_px)) % 360
    directions = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]
    idx = int((angle + 22.5) // 45) % 8
    heading = directions[idx]

    return (vx_mph, vy_mph), speed_mph