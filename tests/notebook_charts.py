"""Handwritten model-response fixtures for offline notebook checks, not LLM benchmarks."""

CV_BARS = '''def plot_data(data, *, title=None, figsize=None):
    import numpy as np
    import matplotlib.pyplot as plt
    highlight = False
    names = list(data)
    positions = np.arange(len(names))
    fig, ax = plt.subplots(figsize=figsize or (11, 5.5))
    colors = ["#087EBC", "#ED8500"]
    labels = ["Accuracy", "ROC-AUC"]
    for i, metric in enumerate(["accuracy", "roc_auc"]):
        means = np.array([np.mean(data[name][metric]) for name in names])
        stds = np.array([np.std(data[name][metric], ddof=1) for name in names])
        bars = ax.bar(positions + (i - 0.5) * 0.36, means, width=0.36,
                      yerr=stds, capsize=4, color=colors[i], label=labels[i],
                      error_kw={"elinewidth": 1.3}, zorder=3)
        for j, bar in enumerate(bars):
            if highlight and np.isclose(means[j], means.max()):
                bar.set_hatch("//")
                bar.set_edgecolor("#172C3E")
                bar.set_linewidth(1.5)
                ax.text(bar.get_x() + bar.get_width() / 2, means[j] + stds[j] + 0.02,
                        f"{means[j]:.3f}", ha="center", va="bottom", fontweight="bold")
            elif highlight:
                bar.set_alpha(0.3)
    handles, legend_labels = ax.get_legend_handles_labels()
    if highlight:
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white",
                                     edgecolor="#172C3E", hatch="//"))
        legend_labels.append("Highest mean per metric")
    ax.set_xticks(positions, names)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("CV score")
    ax.set_title(title or "Model comparison | mean ± 1 sample SD across folds", pad=40)
    ax.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
'''

CV_HIGHLIGHT = CV_BARS.replace("highlight = False", "highlight = True")

ORDERS_DOTS = '''def plot_data(data, *, title=None, figsize=None):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    frame = data.sort_values("week")
    fig, ax = plt.subplots(figsize=figsize or (11, 5.5))
    actuals = frame.dropna(subset=["actual"])
    sns.scatterplot(data=actuals, x="week", y="actual", color="#008C95",
                    s=60, alpha=0.85, label="Observed orders", ax=ax)
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set(xlabel="Week", ylabel="Orders")
    ax.set_title(title or "Weekly orders", pad=18)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig
'''

FORECAST_OVERLAY = '''    forecast = frame.dropna(subset=["forecast"])
    ax.fill_between(frame["week"], frame["lower"], frame["upper"],
                    color="#ED8500", alpha=0.16, label="Supplied interval (synthetic)")
    ax.plot(frame["week"], frame["forecast"], "o--", color="#ED8500", linewidth=2,
            label="Supplied forecast")
    boundary = forecast["week"].iloc[0]
    ax.axvline(boundary, color="#526171", linestyle=":", linewidth=1.2)
    ax.text(boundary, 0.03, " Forecast starts", transform=ax.get_xaxis_transform(),
            color="#526171", fontsize=10)
    outside = frame[(frame["actual"] < frame["lower"]) | (frame["actual"] > frame["upper"])]
    ax.scatter(outside["week"], outside["actual"], marker="D", s=90, color="#BE3455",
               edgecolors="white", linewidths=0.8, label="Outside supplied interval", zorder=5)
    ax.set_title(title or "Weekly orders | supplied forecast and interval", pad=18)
    ax.legend(frameon=False, loc="upper left")'''

ORDERS_FORECAST = ORDERS_DOTS.replace(
    "    fig.tight_layout()", FORECAST_OVERLAY + "\n    fig.tight_layout()"
)

ORDERS_PLOTLY = '''def plot_data(data, *, title=None, figsize=None):
    import plotly.graph_objects as go
    frame = data.sort_values("week")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["week"], y=frame["lower"], mode="lines",
                             line={"width": 0}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=frame["week"], y=frame["upper"], mode="lines",
                             line={"width": 0}, fill="tonexty", fillcolor="rgba(237,133,0,0.16)",
                             name="Supplied interval (synthetic)", hoverinfo="skip"))
    hover = frame[["actual", "forecast", "lower", "upper"]].to_numpy()
    template = ("%{x|%d %b %Y}<br>Actual: %{customdata[0]:,.0f}"
                "<br>Forecast: %{customdata[1]:,.0f}<br>Lower: %{customdata[2]:,.0f}"
                "<br>Upper: %{customdata[3]:,.0f}<extra></extra>")
    fig.add_trace(go.Scatter(x=frame["week"], y=frame["actual"], mode="markers",
                             marker={"color": "#008C95", "size": 8}, name="Observed orders",
                             customdata=hover, hovertemplate=template))
    fig.add_trace(go.Scatter(x=frame["week"], y=frame["forecast"], mode="lines+markers",
                             line={"color": "#ED8500", "dash": "dash"}, connectgaps=False,
                             name="Supplied forecast", customdata=hover, hovertemplate=template))
    outside = frame[(frame["actual"] < frame["lower"]) | (frame["actual"] > frame["upper"])]
    fig.add_trace(go.Scatter(x=outside["week"], y=outside["actual"], mode="markers",
                             marker={"color": "#BE3455", "symbol": "diamond", "size": 10},
                             name="Outside supplied interval",
                             customdata=outside[["actual", "forecast", "lower", "upper"]],
                             hovertemplate=template))
    boundary = frame.dropna(subset=["forecast"])["week"].iloc[0]
    fig.add_vline(x=boundary, line_dash="dot", line_color="#526171")
    fig.update_layout(template="plotly_white", xaxis_title="Week", yaxis_title="Orders",
                       title=title or "Weekly orders | supplied forecast and interval")
    if figsize is not None:
        fig.update_layout(width=figsize[0] * 100, height=figsize[1] * 100)
    return fig
'''
