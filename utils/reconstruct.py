import pandas as pd


def clean_real_points(real_points):
    """
    Sort by time, clamp survival to [0,1], and enforce monotone non-increasing survival.
    """
    if not real_points:
        return []

    real_points = sorted(real_points, key=lambda x: x[0])

    cleaned = []
    last_survival = 1.0

    for i, (t, s) in enumerate(real_points):
        s = max(0.0, min(1.0, s))

        if i == 0:
            cleaned.append((t, s))
            last_survival = s
            continue

        if s > last_survival:
            s = last_survival

        cleaned.append((t, s))
        last_survival = s

    return cleaned


def json_points_to_real_points(points):
    """
    Directly interpret JSON points as (time, survival).

    Rules:
    - x is treated directly as time
    - y is treated directly as survival
      * if y <= 1.5 -> already in 0-1 scale
      * if y > 1.5 -> treated as percentage, divided by 100
    """
    if not points:
        return []

    real_points = []

    for p in points:
        time_val = float(p["x"])
        raw_y = float(p["y"])

        if raw_y <= 1.5:
            survival_val = raw_y
        else:
            survival_val = raw_y / 100.0

        real_points.append((time_val, survival_val))

    return clean_real_points(real_points)


def apply_manual_x_axis(real_points, x_min=0.0, x_max=None, x_tick=None):
    """
    Reassign time values using user-provided x-axis settings.

    Priority:
    - if x_tick is provided, use strict grid: x_min + i * x_tick
    - else if x_max is provided, spread evenly from x_min to x_max
    - else keep original times
    """
    if not real_points:
        return []

    n = len(real_points)

    if x_tick is not None and x_tick > 0:
        return [(x_min + i * x_tick, s) for i, (_, s) in enumerate(real_points)]

    if x_max is not None:
        if n == 1:
            return [(x_min, real_points[0][1])]
        return [
            (x_min + i * (x_max - x_min) / (n - 1), s)
            for i, (_, s) in enumerate(real_points)
        ]

    return real_points


def km_data_to_dataframe(km_data, x_max_override=None, x_tick_override=None):
    """
    Internal dataframe:
    - group
    - point_index
    - time
    - survival
    """
    rows = []

    x_min = km_data.get("x_axis", {}).get("min", 0.0)

    for group in km_data.get("groups", []):
        group_name = group.get("name", "Unknown")
        points = group.get("points", [])

        real_points = json_points_to_real_points(points)
        real_points = apply_manual_x_axis(
            real_points,
            x_min=x_min,
            x_max=x_max_override,
            x_tick=x_tick_override
        )

        for idx, (time_val, survival_val) in enumerate(real_points):
            rows.append({
                "group": group_name,
                "point_index": idx,
                "time": float(time_val),
                "survival": float(survival_val)
            })

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(["group", "time"]).reset_index(drop=True)

    return df


def validate_survival_dataframe(df):
    """
    Validate internal dataframe:
    - time non-decreasing
    - survival monotone non-increasing
    - survival in [0,1]
    """
    if df.empty:
        return True, ""

    for group_name, g in df.groupby("group"):
        g = g.sort_values("time").reset_index(drop=True)

        times = g["time"].tolist()
        survs = g["survival"].tolist()

        for i in range(len(survs)):
            s = float(survs[i])

            if s < 0 or s > 1:
                return False, f"{group_name}: survival must stay between 0 and 1."

            if i > 0 and s > float(survs[i - 1]):
                return False, f"{group_name}: row {i} survival must be <= previous survival."

            if i < len(survs) - 1 and s < float(survs[i + 1]):
                return False, f"{group_name}: row {i} survival must be >= next survival."

            if i > 0 and float(times[i]) < float(times[i - 1]):
                return False, f"{group_name}: time must be non-decreasing."

    return True, ""


def dataframe_to_group_real_points(df):
    """
    Convert internal dataframe back into dict of group -> [(time, survival), ...]
    """
    result = {}

    if df.empty:
        return result

    for group_name, g in df.groupby("group"):
        g = g.sort_values("time").reset_index(drop=True)
        pts = list(zip(g["time"].astype(float), g["survival"].astype(float)))
        result[group_name] = clean_real_points(pts)

    return result


def real_points_to_pseudo_dataset(real_points, group_name, initial_n=100):
    """
    Convert (time, survival) points into a pseudo individual-level dataset for log-rank.
    """
    if not real_points:
        return pd.DataFrame(columns=["time", "event", "group"])

    rows = []

    if real_points[0][1] < 0.999:
        real_points = [(0.0, 1.0)] + real_points

    prev_survival = real_points[0][1]

    for i in range(1, len(real_points)):
        current_time, current_survival = real_points[i]

        drop = max(0.0, prev_survival - current_survival)
        n_events = round(initial_n * drop)

        for _ in range(n_events):
            rows.append({
                "time": current_time,
                "event": 1,
                "group": group_name
            })

        prev_survival = current_survival

    total_events = len(rows)
    remaining = max(0, initial_n - total_events)
    final_time = real_points[-1][0]

    for _ in range(remaining):
        rows.append({
            "time": final_time,
            "event": 0,
            "group": group_name
        })

    return pd.DataFrame(rows)


def build_logrank_dataframe_from_points(group_points, sample_sizes=None):
    """
    Build pseudo log-rank dataset from a dict:
    {group_name: [(time, survival), ...]}
    """
    if sample_sizes is None:
        sample_sizes = {}

    dfs = []

    for group_name, real_points in group_points.items():
        initial_n = sample_sizes.get(group_name, 100)

        df_group = real_points_to_pseudo_dataset(
            real_points=real_points,
            group_name=group_name,
            initial_n=initial_n
        )

        dfs.append(df_group)

    if dfs:
        return pd.concat(dfs, ignore_index=True)

    return pd.DataFrame(columns=["time", "event", "group"])


def build_logrank_dataframe(km_data, sample_sizes=None, x_max_override=None, x_tick_override=None):
    df_points = km_data_to_dataframe(
        km_data,
        x_max_override=x_max_override,
        x_tick_override=x_tick_override
    )
    group_points = dataframe_to_group_real_points(df_points)
    return build_logrank_dataframe_from_points(group_points, sample_sizes=sample_sizes)


def _group_points_to_summary_df(real_points, initial_n=100):
    """
    Convert (time, survival) points into a display summary table:
    - time
    - event_count
    - survival
    """
    if not real_points:
        return pd.DataFrame(columns=["time", "event_count", "survival"])

    real_points = clean_real_points(real_points)

    if real_points[0][1] < 0.999:
        real_points = [(0.0, 1.0)] + real_points

    rows = []
    prev_survival = real_points[0][1]

    rows.append({
        "time": float(real_points[0][0]),
        "event_count": 0,
        "survival": float(real_points[0][1])
    })

    for i in range(1, len(real_points)):
        current_time, current_survival = real_points[i]
        drop = max(0.0, prev_survival - current_survival)
        event_count = round(initial_n * drop)

        rows.append({
            "time": float(current_time),
            "event_count": int(event_count),
            "survival": float(current_survival)
        })

        prev_survival = current_survival

    return pd.DataFrame(rows)


def km_data_to_group_summary_tables(km_data, sample_sizes=None, x_max_override=None, x_tick_override=None):
    """
    Create one summary dataframe per group from JSON:
    columns = time, event_count, survival
    """
    if sample_sizes is None:
        sample_sizes = {}

    df_points = km_data_to_dataframe(
        km_data,
        x_max_override=x_max_override,
        x_tick_override=x_tick_override
    )
    group_points = dataframe_to_group_real_points(df_points)

    result = {}
    for group_name, real_points in group_points.items():
        initial_n = sample_sizes.get(group_name, 100)
        result[group_name] = _group_points_to_summary_df(real_points, initial_n=initial_n)

    return result


def edited_dataframe_to_group_summary_tables(df, sample_sizes=None):
    """
    Create one summary dataframe per group from edited dataframe:
    columns = time, event_count, survival
    """
    if sample_sizes is None:
        sample_sizes = {}

    group_points = dataframe_to_group_real_points(df)
    result = {}

    for group_name, real_points in group_points.items():
        initial_n = sample_sizes.get(group_name, 100)
        result[group_name] = _group_points_to_summary_df(real_points, initial_n=initial_n)

    return result


def validate_group_summary_tables(group_tables):
    """
    Validate editable grouped summary tables.
    Required columns:
    - time
    - survival

    Conditions:
    - time non-decreasing
    - survival monotone non-increasing
    - survival in [0,1]
    """
    if not group_tables:
        return True, ""

    for group_name, df in group_tables.items():
        if df.empty:
            continue

        g = df.sort_values("time").reset_index(drop=True)
        times = g["time"].tolist()
        survs = g["survival"].tolist()

        for i in range(len(survs)):
            s = float(survs[i])

            if s < 0 or s > 1:
                return False, f"{group_name}: survival must stay between 0 and 1."

            if i > 0 and s > float(survs[i - 1]):
                return False, f"{group_name}: row {i} survival must be <= previous survival."

            if i < len(survs) - 1 and s < float(survs[i + 1]):
                return False, f"{group_name}: row {i} survival must be >= next survival."

            if i > 0 and float(times[i]) < float(times[i - 1]):
                return False, f"{group_name}: time must be non-decreasing."

    return True, ""


def get_group_survival_bounds(group_tables):
    """
    Add lower/upper bounds to each group summary table.
    Bounds are based on monotone non-increasing survival.
    """
    result = {}

    for group_name, df in group_tables.items():
        if df.empty:
            result[group_name] = df.copy()
            continue

        out = df.copy().sort_values("time").reset_index(drop=True)
        survs = out["survival"].tolist()

        uppers = []
        lowers = []

        for i in range(len(survs)):
            prev_surv = survs[i - 1] if i > 0 else 1.0
            next_surv = survs[i + 1] if i < len(survs) - 1 else 0.0

            upper = min(1.0, prev_surv)
            lower = max(0.0, next_surv)

            uppers.append(upper)
            lowers.append(lower)

        out["lower_bound"] = lowers
        out["upper_bound"] = uppers
        result[group_name] = out

    return result


def group_summary_tables_to_internal_dataframe(group_tables):
    """
    Convert editable group summary tables back to internal dataframe:
    - group
    - point_index
    - time
    - survival
    """
    rows = []

    for group_name, df in group_tables.items():
        if df.empty:
            continue

        g = df.sort_values("time").reset_index(drop=True)

        for idx, row in g.iterrows():
            rows.append({
                "group": group_name,
                "point_index": idx,
                "time": float(row["time"]),
                "survival": float(row["survival"])
            })

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(["group", "time"]).reset_index(drop=True)

    return out