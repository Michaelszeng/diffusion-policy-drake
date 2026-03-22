"""
Numpy port of ManiSkill's pseudo_render_intersection for the small_t_pusher geometry.

The small_t_pusher (from arbitrary_shape_pickles/small_t_pusher.pkl) has:
  Box 1 (horizontal bar): size [0.1651, 0.04064], centered at (0, 0) in body frame
  Box 2 (vertical stem):  size [0.04064, 0.12447], centered at (0, -0.08255) in body frame

Drake reports the T pose in the body frame (SDF model origin = center of box 1,
since center_of_mass=null in physical_properties).

The overlap is computed as: intersection_area / goal_area, both measured on a 64x64
pixel grid spanning ±uv_half_width meters.
"""

import numpy as np

# small_t_pusher geometry in body frame (origin = center of horizontal bar)
_BOX1_HALF = np.array([0.1651 / 2, 0.04064 / 2])  # (0.08255, 0.02032)
_BOX1_CENTER = np.array([0.0, 0.0])
_BOX2_HALF = np.array([0.04064 / 2, 0.12447 / 2])  # (0.02032, 0.06224)
_BOX2_CENTER = np.array([0.0, -0.08255])


def build_tee_mask(res: int = 64, uv_half_width: float = 0.15) -> np.ndarray:
    """
    Build a (res, res) binary mask of the canonical T shape in its own body frame.
    The mask is in image coordinates: row=0 is y=+uv_half_width, col=0 is x=-uv_half_width.
    """
    # Build UV grid: each pixel's (x, y) world coordinate
    # Shape: (res, res) arrays for x and y
    lin = (np.arange(res) + 0.5) / res * 2 * uv_half_width - uv_half_width
    xs = lin[None, :].repeat(res, axis=0)  # (res, res): col index → x
    ys = (lin[::-1])[:, None].repeat(res, axis=1)  # (res, res): row index → y (flipped)

    in_box1 = (
        (xs >= _BOX1_CENTER[0] - _BOX1_HALF[0])
        & (xs <= _BOX1_CENTER[0] + _BOX1_HALF[0])
        & (ys >= _BOX1_CENTER[1] - _BOX1_HALF[1])
        & (ys <= _BOX1_CENTER[1] + _BOX1_HALF[1])
    )
    in_box2 = (
        (xs >= _BOX2_CENTER[0] - _BOX2_HALF[0])
        & (xs <= _BOX2_CENTER[0] + _BOX2_HALF[0])
        & (ys >= _BOX2_CENTER[1] - _BOX2_HALF[1])
        & (ys <= _BOX2_CENTER[1] + _BOX2_HALF[1])
    )
    return (in_box1 | in_box2).astype(np.float32)


# Pre-build canonical mask (reused across all calls)
_TEE_MASK = build_tee_mask(res=64, uv_half_width=0.15)
_GOAL_AREA = float(_TEE_MASK.sum())


def _build_homo_transform(x: float, y: float, theta: float) -> np.ndarray:
    """Build a 3x3 2D homogeneous rigid-body transform."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [c, -s, x],
            [s, c, y],
            [0, 0, 1],
        ]
    )


def compute_tee_overlap(
    tee_x: float,
    tee_y: float,
    tee_theta: float,
    goal_x: float,
    goal_y: float,
    goal_theta: float,
    res: int = 64,
    uv_half_width: float = 0.15,
) -> float:
    """
    Compute the fraction of the goal T's area that is covered by the actual T.

    Returns a value in [0, 1], where 1.0 = perfect overlap.

    Uses the same 'pseudo-render' algorithm as ManiSkill's pseudo_render_intersection:
    1. Express the actual T's body-frame pixels in the goal T's frame.
    2. Count how many land inside the goal T mask.
    3. Divide by the goal area.
    """
    # Use cached mask/area if resolution matches, else build fresh
    if res == 64 and uv_half_width == 0.15:
        tee_mask = _TEE_MASK
        goal_area = _GOAL_AREA
    else:
        tee_mask = build_tee_mask(res, uv_half_width)
        goal_area = float(tee_mask.sum())

    if goal_area == 0:
        return 0.0

    # --- Transform: tee body frame → world → goal body frame ---
    T_tee_to_world = _build_homo_transform(tee_x, tee_y, tee_theta)
    T_goal_to_world = _build_homo_transform(goal_x, goal_y, goal_theta)
    T_world_to_goal = np.linalg.inv(T_goal_to_world)
    T_tee_to_goal = T_world_to_goal @ T_tee_to_world  # (3, 3)

    # --- Get (x, y) coordinates of all pixels in the canonical tee_mask ---
    # Pixel indices where the canonical T occupies
    rows, cols = np.where(tee_mask > 0)  # (N,) each
    # Convert pixel indices to world coordinates in tee body frame
    # col → x, row → y (with y-flip)
    x_tee = (cols + 0.5) / res * 2 * uv_half_width - uv_half_width
    y_tee = uv_half_width - (rows + 0.5) / res * 2 * uv_half_width

    # Homogeneous coords: (3, N)
    N = len(x_tee)
    pts_homo = np.ones((3, N))
    pts_homo[0] = x_tee
    pts_homo[1] = y_tee

    # Transform to goal frame
    pts_goal = T_tee_to_goal @ pts_homo  # (3, N)
    x_goal = pts_goal[0]
    y_goal = pts_goal[1]

    # Convert goal-frame coordinates to pixel indices in the goal mask
    col_goal = np.floor((x_goal + uv_half_width) / (2 * uv_half_width) * res).astype(int)
    row_goal = np.floor((uv_half_width - y_goal) / (2 * uv_half_width) * res).astype(int)

    # Filter out-of-bounds
    valid = (col_goal >= 0) & (col_goal < res) & (row_goal >= 0) & (row_goal < res)
    col_goal = col_goal[valid]
    row_goal = row_goal[valid]

    # Build rendered image of tee in goal frame
    rendered = np.zeros((res, res), dtype=np.float32)
    rendered[row_goal, col_goal] = 1.0

    # Intersection / goal_area
    intersection = float((rendered.astype(bool) & tee_mask.astype(bool)).sum())
    return intersection / goal_area
