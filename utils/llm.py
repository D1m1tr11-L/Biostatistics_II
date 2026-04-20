import base64
import json
from statistics import mean
from openai import OpenAI

client = OpenAI()


def encode_image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def _build_prompt(x_max=None, x_tick=None, force_last_zero_groups=None):
    if force_last_zero_groups is None:
        force_last_zero_groups = []

    strict_x_mode = (x_max is not None) and (x_tick is not None)

    constraint_lines = []

    if strict_x_mode:
        n_points = int(round(x_max / x_tick)) + 1
        constraint_lines.extend([
            "- STRICT GRID MODE IS ACTIVE.",
            f"- The x-axis maximum value is EXACTLY {x_max}.",
            f"- The x-axis tick interval is EXACTLY {x_tick}.",
            f"- Therefore, the exact number of x positions is {n_points}.",
            "- You MUST return exactly one point per group at each x position.",
            "- You MUST NOT omit any x position.",
            "- You MUST NOT infer a different x-axis maximum or a different x-axis tick interval.",
            "- The x positions correspond exactly to: 0, tick, 2*tick, ..., x_max."
        ])
    else:
        constraint_lines.extend([
            "- AUTO GRID MODE IS ACTIVE.",
            "- Infer x-axis maximum and x-axis tick interval from the image.",
            "- Then determine the exact number of x positions as x_max / x_tick + 1.",
            "- After inferring the x-axis grid, you MUST return exactly one point per group at each x position.",
            "- You MUST NOT omit any x position."
        ])

    if force_last_zero_groups:
        joined = ", ".join(force_last_zero_groups)
        constraint_lines.append(
            f"- The following groups must have final survival equal to 0 at the last x position: {joined}."
        )

    constraint_lines.extend([
        "- Kaplan-Meier survival curves must be monotonically non-increasing from left to right.",
        "- Do NOT output points that imply increasing survival over time.",
        "- The first point of each group should correspond to early time / high survival.",
        "- The curve should follow the stepwise Kaplan-Meier structure, not a diagonal trend line.",
        "- If a group's point at a required x position is visually unclear:",
        "  1. If the curve has already dropped to zero by then, set that point to survival 0.",
        "  2. Otherwise, if curves overlap at that x position, set that group's point to the same y as the overlapping group.",
        "  3. Do NOT leave the point missing.",
        "- Return x_axis.tick explicitly in the JSON."
    ])

    constraint_text = "\n".join(constraint_lines)

    prompt = f"""
You are analyzing a Kaplan-Meier survival plot image.

Return ONLY valid JSON.
Do not include markdown.
Do not include explanations.

IMPORTANT CONSTRAINTS:
{constraint_text}

Task:
1. Identify the main plotting box:
   - x_left
   - x_right
   - y_top
   - y_bottom

2. Identify x-axis range:
   - min
   - max
   - tick
   - unit

3. Identify y-axis range:
   - min
   - max

4. Identify exactly up to 2 survival groups from the legend if visible.

5. For each group, return EXACTLY one point for each x-grid position.
   - The point list length for every group must be identical.
   - Each group must have a point at every required x position.
   - Each point must have integer x and y coordinates.
   - The points must be ordered left to right.

Return this JSON structure exactly:

{{
  "plot_box": {{
    "x_left": 0,
    "x_right": 0,
    "y_top": 0,
    "y_bottom": 0
  }},
  "x_axis": {{
    "min": 0,
    "max": 0,
    "tick": 0,
    "unit": "months"
  }},
  "y_axis": {{
    "min": 0.0,
    "max": 1.0
  }},
  "groups": [
    {{
      "name": "Group A",
      "color": "blue",
      "points": [
        {{"x": 0, "y": 0}}
      ]
    }},
    {{
      "name": "Group B",
      "color": "red",
      "points": [
        {{"x": 0, "y": 0}}
      ]
    }}
  ]
}}
"""
    return prompt


def _postprocess_axis(data, x_max=None, x_tick=None):
    if "x_axis" not in data:
        data["x_axis"] = {}

    data["x_axis"].setdefault("min", 0)
    data["x_axis"].setdefault("unit", "months")

    if x_max is not None:
        data["x_axis"]["max"] = x_max

    if x_tick is not None:
        data["x_axis"]["tick"] = x_tick

    # If auto mode and LLM forgot tick, infer it from max and point count
    if data["x_axis"].get("tick") in [None, 0]:
        groups = data.get("groups", [])
        if groups and groups[0].get("points"):
            n = len(groups[0]["points"])
            x_max_val = data["x_axis"].get("max", None)
            if x_max_val is not None and n > 1:
                data["x_axis"]["tick"] = x_max_val / (n - 1)

    return data


def _force_last_zero(data, force_last_zero_groups=None):
    if force_last_zero_groups is None:
        force_last_zero_groups = []

    if not force_last_zero_groups:
        return data

    plot_box = data.get("plot_box", {})
    y_bottom = plot_box.get("y_bottom", None)

    if y_bottom is None:
        return data

    for group in data.get("groups", []):
        if group.get("name") in force_last_zero_groups and group.get("points"):
            group["points"][-1]["y"] = int(y_bottom)

    return data


def extract_km_data_from_image(
    image_bytes: bytes,
    x_max=None,
    x_tick=None,
    force_last_zero_groups=None
):
    if force_last_zero_groups is None:
        force_last_zero_groups = []

    image_b64 = encode_image_to_base64(image_bytes)
    prompt = _build_prompt(
        x_max=x_max,
        x_tick=x_tick,
        force_last_zero_groups=force_last_zero_groups
    )

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        top_p=1,
        seed=12345,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "developer",
                "content": (
                    "Return valid JSON only. "
                    "Always return a complete x-grid with one point per group per x position. "
                    "If x_max and x_tick are supplied, treat them as exact."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]
    )

    content = response.choices[0].message.content
    data = json.loads(content)
    system_fingerprint = getattr(response, "system_fingerprint", None)

    data = _postprocess_axis(data, x_max=x_max, x_tick=x_tick)
    data = _force_last_zero(data, force_last_zero_groups=force_last_zero_groups)

    return data, system_fingerprint


def _average_numeric_dict(dict_list, keys):
    averaged = {}
    for key in keys:
        values = [d[key] for d in dict_list if key in d]
        averaged[key] = int(round(mean(values))) if values else 0
    return averaged


def _average_axis_dict(dict_list, keys):
    averaged = {}
    for key in keys:
        values = [d[key] for d in dict_list if key in d]
        averaged[key] = mean(values) if values else 0
    return averaged


def extract_km_data_with_sampling(
    image_bytes: bytes,
    n_samples: int = 20,
    x_max=None,
    x_tick=None,
    force_last_zero_groups=None
):
    """
    Multiple LLM runs:
    - x is averaged across runs
    - y is taken from the first successful run
    - x grid is strict if x_max and x_tick are provided
    - otherwise x grid is inferred and then applied strictly
    """
    if force_last_zero_groups is None:
        force_last_zero_groups = []

    successful_runs = []
    fingerprints = []

    for _ in range(n_samples):
        try:
            data, fp = extract_km_data_from_image(
                image_bytes=image_bytes,
                x_max=x_max,
                x_tick=x_tick,
                force_last_zero_groups=force_last_zero_groups
            )
            successful_runs.append(data)
            fingerprints.append(fp)
        except Exception:
            continue

    if not successful_runs:
        raise ValueError("All LLM extraction runs failed.")

    base = successful_runs[0]

    # plotting box
    plot_boxes = [run["plot_box"] for run in successful_runs if "plot_box" in run]
    avg_plot_box = _average_numeric_dict(
        plot_boxes,
        ["x_left", "x_right", "y_top", "y_bottom"]
    )

    # axis
    x_axes = [run["x_axis"] for run in successful_runs if "x_axis" in run]
    avg_x_axis = _average_axis_dict(x_axes, ["min", "max", "tick"])
    avg_x_axis["unit"] = x_axes[0].get("unit", "months") if x_axes else "months"

    y_axes = [run["y_axis"] for run in successful_runs if "y_axis" in run]
    avg_y_axis = _average_axis_dict(y_axes, ["min", "max"])

    # Hard override if user provided
    if x_max is not None:
        avg_x_axis["max"] = x_max
    if x_tick is not None:
        avg_x_axis["tick"] = x_tick

    averaged_groups = []
    base_groups = base.get("groups", [])[:2]

    for group_idx, base_group in enumerate(base_groups):
        group_name = base_group.get("name", f"Group {group_idx + 1}")
        group_color = base_group.get("color", "")

        all_group_points = []
        for run in successful_runs:
            groups = run.get("groups", [])
            if len(groups) > group_idx:
                pts = groups[group_idx].get("points", [])
                if pts:
                    all_group_points.append(pts)

        if not all_group_points:
            averaged_groups.append({
                "name": group_name,
                "color": group_color,
                "points": []
            })
            continue

        min_len = min(len(pts) for pts in all_group_points)
        aligned_runs = [pts[:min_len] for pts in all_group_points]

        averaged_points = []
        for point_idx in range(min_len):
            xs = [run_pts[point_idx]["x"] for run_pts in aligned_runs]
            x_avg = int(round(mean(xs)))

            # y from first successful run to preserve step structure
            y_fixed = aligned_runs[0][point_idx]["y"]

            averaged_points.append({
                "x": x_avg,
                "y": y_fixed
            })

        averaged_groups.append({
            "name": group_name,
            "color": group_color,
            "points": averaged_points
        })

    final_data = {
        "plot_box": avg_plot_box,
        "x_axis": avg_x_axis,
        "y_axis": avg_y_axis,
        "groups": averaged_groups
    }

    final_data = _postprocess_axis(final_data, x_max=x_max, x_tick=x_tick)
    final_data = _force_last_zero(final_data, force_last_zero_groups=force_last_zero_groups)

    meta = {
        "requested_samples": n_samples,
        "successful_samples": len(successful_runs),
        "fingerprints": fingerprints,
        "manual_x_max": x_max,
        "manual_x_tick": x_tick,
        "forced_zero_groups": force_last_zero_groups,
        "strict_grid_mode": (x_max is not None and x_tick is not None)
    }

    return final_data, meta