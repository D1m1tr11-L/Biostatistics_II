import io
from PIL import Image
import matplotlib.pyplot as plt


def _fraction_in_range(values, low, high, tol=0):
    if not values:
        return 0.0
    count = sum((low - tol) <= v <= (high + tol) for v in values)
    return count / len(values)


def _is_mostly_nonincreasing(values, tol=1e-6):
    if len(values) < 2:
        return False
    noninc = sum(values[i] <= values[i - 1] + tol for i in range(1, len(values)))
    return noninc / (len(values) - 1) > 0.7


def _is_mostly_nondecreasing(values, tol=1e-6):
    if len(values) < 2:
        return False
    nondec = sum(values[i] >= values[i - 1] - tol for i in range(1, len(values)))
    return nondec / (len(values) - 1) > 0.7


def _detect_x_mode(points, km_data):
    if not points:
        return "pixel"

    plot_box = km_data.get("plot_box", {})
    x_axis = km_data.get("x_axis", {})

    x_left = plot_box.get("x_left", 0)
    x_right = plot_box.get("x_right", 0)

    x_min = x_axis.get("min", 0)
    x_max = x_axis.get("max", 0)

    xs = [p.get("x", 0) for p in points]

    frac_pixel = _fraction_in_range(xs, x_left, x_right, tol=10)
    frac_data = _fraction_in_range(xs, x_min, x_max, tol=1)

    if frac_data > 0.8 and frac_pixel < 0.5:
        return "data"
    return "pixel"


def _detect_y_mode(points, km_data, x_mode):
    """
    Detect whether y values are:
    - pixel coordinates
    - data/survival in 0-1
    - data/survival in 0-100

    Important fix:
    If x is already data and y decreases with x, that strongly suggests
    y is survival data, not pixel coordinates.
    """
    if not points:
        return "pixel"

    plot_box = km_data.get("plot_box", {})
    y_axis = km_data.get("y_axis", {})

    y_top = plot_box.get("y_top", 0)
    y_bottom = plot_box.get("y_bottom", 0)

    y_min = y_axis.get("min", 0.0)
    y_max = y_axis.get("max", 1.0)

    ys = [p.get("y", 0) for p in points]

    frac_pixel = _fraction_in_range(ys, y_top, y_bottom, tol=10)
    frac_data01 = _fraction_in_range(ys, y_min, y_max, tol=0.15)
    frac_data100 = _fraction_in_range(ys, 0, 100, tol=5)

    # Direction-based disambiguation
    # For a decreasing KM curve:
    # - survival data should mostly decrease
    # - pixel y should mostly increase (downward on image)
    mostly_noninc = _is_mostly_nonincreasing(ys)
    mostly_nondec = _is_mostly_nondecreasing(ys)

    # If x is data and y looks like 0-100 percentages and is decreasing,
    # prefer data_100 even if values also lie inside the pixel box range.
    if x_mode == "data" and frac_data100 > 0.8 and mostly_noninc:
        return "data_100"

    # If x is data and y looks like 0-1 survival and is decreasing
    if x_mode == "data" and frac_data01 > 0.8 and mostly_noninc:
        return "data_01"

    # If y behaves like pixel coordinates (typically increasing downward)
    if frac_pixel > 0.8 and mostly_nondec:
        return "pixel"

    # Fallbacks
    if frac_data01 > 0.8 and frac_pixel < 0.5:
        return "data_01"

    if frac_data100 > 0.8 and frac_pixel < 0.5:
        return "data_100"

    return "pixel"


def _data_x_to_pixel(x, km_data):
    plot_box = km_data["plot_box"]
    x_axis = km_data["x_axis"]

    x_left = plot_box["x_left"]
    x_right = plot_box["x_right"]

    x_min = x_axis.get("min", 0)
    x_max = x_axis.get("max", 1)

    if x_max == x_min:
        return x_left

    return x_left + (x - x_min) / (x_max - x_min) * (x_right - x_left)


def _data_y_to_pixel(y, km_data, y_mode="data_01"):
    plot_box = km_data["plot_box"]
    y_axis = km_data["y_axis"]

    y_top = plot_box["y_top"]
    y_bottom = plot_box["y_bottom"]

    y_min = y_axis.get("min", 0.0)
    y_max = y_axis.get("max", 1.0)

    if y_mode == "data_100":
        y = y / 100.0

    if y_max == y_min:
        return y_bottom

    return y_top + (y_max - y) / (y_max - y_min) * (y_bottom - y_top)


def _normalize_points_to_pixel(points, km_data):
    if not points:
        return [], "pixel", "pixel"

    x_mode = _detect_x_mode(points, km_data)
    y_mode = _detect_y_mode(points, km_data, x_mode=x_mode)

    normalized = []
    for p in points:
        raw_x = p["x"]
        raw_y = p["y"]

        px = _data_x_to_pixel(raw_x, km_data) if x_mode == "data" else raw_x

        if y_mode in ["data_01", "data_100"]:
            py = _data_y_to_pixel(raw_y, km_data, y_mode=y_mode)
        else:
            py = raw_y

        normalized.append({"x": px, "y": py})

    return normalized, x_mode, y_mode


def plot_points_on_image(image_bytes: bytes, km_data: dict):
    image = Image.open(io.BytesIO(image_bytes))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image)

    plot_box = km_data.get("plot_box", {})
    if plot_box:
        x_left = plot_box.get("x_left", 0)
        x_right = plot_box.get("x_right", 0)
        y_top = plot_box.get("y_top", 0)
        y_bottom = plot_box.get("y_bottom", 0)

        rect_x = [x_left, x_right, x_right, x_left, x_left]
        rect_y = [y_top, y_top, y_bottom, y_bottom, y_top]
        ax.plot(rect_x, rect_y, linestyle="--", label="Plot Box")

    groups = km_data.get("groups", [])
    for group in groups:
        pts = group.get("points", [])
        if not pts:
            continue

        pts_pixel, x_mode, y_mode = _normalize_points_to_pixel(pts, km_data)

        xs = [p["x"] for p in pts_pixel]
        ys = [p["y"] for p in pts_pixel]

        label = group.get("name", "Unknown")
        label = f"{label} (x:{x_mode}, y:{y_mode})"

        ax.plot(xs, ys, marker="o", linestyle="-", label=label)

    ax.set_title("LLM-Extracted Curve Points")
    ax.legend()
    ax.set_axis_off()

    return fig