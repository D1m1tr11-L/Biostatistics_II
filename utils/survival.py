from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
from scipy.stats import chi2


def format_p_value(p_value: float) -> str:
    """
    Format p-value:
    - if p == 0 or p < 0.0001, show <0.0001
    - otherwise show the true value
    """
    p_value = float(p_value)

    if p_value == 0 or p_value < 0.0001:
        return "<0.0001"

    return f"{p_value:.10g}"


def run_km_analysis_from_dataframe(df, alpha=0.05):
    """
    Run KM and log-rank test from a dataframe with columns:
    time, event, group

    Returns
    -------
    dict with:
    - plot
    - p_value
    - p_value_display
    - test_statistic
    - critical_value
    - alpha
    """
    groups = df["group"].unique()

    if len(groups) != 2:
        raise ValueError("Exactly two groups are required for log-rank test.")

    g1, g2 = groups[0], groups[1]

    df1 = df[df["group"] == g1]
    df2 = df[df["group"] == g2]

    kmf_1 = KaplanMeierFitter()
    kmf_2 = KaplanMeierFitter()

    fig, ax = plt.subplots()

    kmf_1.fit(df1["time"], df1["event"], label=g1)
    kmf_1.plot(ax=ax)

    kmf_2.fit(df2["time"], df2["event"], label=g2)
    kmf_2.plot(ax=ax)

    ax.set_title("Reconstructed Kaplan-Meier Curve")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")

    result = logrank_test(
        df1["time"],
        df2["time"],
        event_observed_A=df1["event"],
        event_observed_B=df2["event"]
    )

    test_statistic = float(result.test_statistic)
    p_value = float(result.p_value)

    # For two-group log-rank test, df = 1
    critical_value = float(chi2.ppf(1 - alpha, df=1))

    return {
        "plot": fig,
        "p_value": p_value,
        "p_value_display": format_p_value(p_value),
        "test_statistic": test_statistic,
        "critical_value": critical_value,
        "alpha": alpha
    }