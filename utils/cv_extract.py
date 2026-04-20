import cv2
import numpy as np


def extract_curve_mask_hsv(img_bgr, lower_hsv, upper_hsv):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(lower_hsv, dtype=np.uint8)
    upper = np.array(upper_hsv, dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


def crop_with_plot_box(img_bgr, plot_box):
    x0 = max(0, int(plot_box["x_left"]))
    y0 = max(0, int(plot_box["y_top"]))
    x1 = min(img_bgr.shape[1], int(plot_box["x_right"]) + 1)
    y1 = min(img_bgr.shape[0], int(plot_box["y_bottom"]) + 1)

    cropped = img_bgr[y0:y1, x0:x1].copy()
    return cropped


def mask_to_centerline(mask):
    pts = []
    h, w = mask.shape

    for x in range(w):
        ys = np.where(mask[:, x] > 0)[0]
        if len(ys) > 0:
            y = int(np.median(ys))
            pts.append((x, y))

    return pts


def interpolate_missing_columns(points):
    if not points:
        return []

    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)

    full_x = np.arange(int(xs.min()), int(xs.max()) + 1)
    full_y = np.interp(full_x, xs, ys)

    return list(zip(full_x.astype(int), full_y.astype(float)))


def median_smooth(y_values, k=5):
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1

    half = k // 2
    out = []

    for i in range(len(y_values)):
        left = max(0, i - half)
        right = min(len(y_values), i + half + 1)
        out.append(float(np.median(y_values[left:right])))

    return np.array(out)


def enforce_km_step_monotonicity(points):
    if not points:
        return []

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    corrected = []
    last_y = ys[0]

    for y in ys:
        if y < last_y:
            y = last_y
        corrected.append(y)
        last_y = y

    return list(zip(xs, corrected))


def detect_key_step_points(points, min_drop_pixels=4, min_x_gap=10):
    if not points:
        return []

    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])

    key_points = [(xs[0], ys[0])]
    last_step_x = xs[0]
    baseline_y = ys[0]

    for i in range(1, len(points)):
        dx = xs[i] - last_step_x
        dy = ys[i] - baseline_y

        if dx >= min_x_gap and dy >= min_drop_pixels:
            key_points.append((xs[i], ys[i]))
            last_step_x = xs[i]
            baseline_y = ys[i]

    if key_points[-1][0] != xs[-1]:
        key_points.append((xs[-1], ys[-1]))

    return key_points


def _curve_points_to_global(points_local, plot_box):
    x0 = int(plot_box["x_left"])
    y0 = int(plot_box["y_top"])

    out = []
    for x, y in points_local:
        out.append({"x": int(round(x + x0)), "y": int(round(y + y0))})
    return out


def extract_km_data_from_image_cv_with_params(
    image_bytes: bytes,
    cv_params: dict,
    min_drop_pixels: int = 4,
    min_x_gap: int = 10,
    smooth_kernel: int = 5,
):
    """
    Use LLM-read parameters to drive CV extraction.
    """
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode uploaded image.")

    plot_box = cv_params["plot_box"]
    cropped_bgr = crop_with_plot_box(img_bgr, plot_box)

    groups = []
    debug = []

    for spec in cv_params.get("groups", [])[:2]:
        lower_hsv = tuple(spec["lower_hsv"])
        upper_hsv = tuple(spec["upper_hsv"])

        mask = extract_curve_mask_hsv(cropped_bgr, lower_hsv, upper_hsv)
        raw_points = mask_to_centerline(mask)

        if not raw_points:
            groups.append({
                "name": spec["name"],
                "color": spec.get("color_label", ""),
                "points": []
            })
            debug.append({
                "group": spec["name"],
                "n_raw_points": 0,
                "n_key_points": 0
            })
            continue

        dense_points = interpolate_missing_columns(raw_points)

        xs = [p[0] for p in dense_points]
        ys = [p[1] for p in dense_points]
        ys_smooth = median_smooth(ys, k=smooth_kernel)

        smoothed_points = list(zip(xs, ys_smooth))
        monotone_points = enforce_km_step_monotonicity(smoothed_points)

        key_points_local = detect_key_step_points(
            monotone_points,
            min_drop_pixels=min_drop_pixels,
            min_x_gap=min_x_gap
        )

        key_points_global = _curve_points_to_global(key_points_local, plot_box)

        groups.append({
            "name": spec["name"],
            "color": spec.get("color_label", ""),
            "points": key_points_global
        })

        debug.append({
            "group": spec["name"],
            "n_raw_points": len(raw_points),
            "n_key_points": len(key_points_global),
            "lower_hsv": lower_hsv,
            "upper_hsv": upper_hsv
        })

    km_data = {
        "plot_box": cv_params["plot_box"],
        "x_axis": cv_params["x_axis"],
        "y_axis": cv_params["y_axis"],
        "groups": groups
    }

    meta = {
        "mode": "llm_plus_cv",
        "debug": debug
    }

    return km_data, meta