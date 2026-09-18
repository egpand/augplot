"""Handwritten model-response fixtures for offline notebook checks, not LLM benchmarks."""

PENGUIN_SCATTER = '''def plot_data(data, *, title=None, figsize=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    frame = data.dropna(subset=["bill_length_mm", "bill_depth_mm", "species"])
    fig, ax = plt.subplots(figsize=figsize or (9, 5.5))
    sns.scatterplot(data=frame, x="bill_length_mm", y="bill_depth_mm", hue="species", ax=ax)
    ax.set(xlabel="Bill length (mm)", ylabel="Bill depth (mm)")
    ax.set_title(title or "Penguin bill measurements by species")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig
'''

PENGUIN_FACETS = '''def plot_data(data, *, title=None, figsize=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    frame = data.dropna(subset=["bill_length_mm", "bill_depth_mm", "species", "sex"])
    species = list(dict.fromkeys(frame["species"]))
    width = (figsize or (12, 4.5))[0]
    height = (figsize or (12, 4.5))[1]
    fig, axes = plt.subplots(1, len(species), figsize=(width, height), sharex=True, sharey=True)
    axes = [axes] if len(species) == 1 else axes
    markers = {"Male": "o", "Female": "s"}
    colors = {"Male": "#087EBC", "Female": "#ED8500"}
    for ax, species_name in zip(axes, species, strict=True):
        subset = frame[frame["species"] == species_name]
        sns.scatterplot(data=subset, x="bill_length_mm", y="bill_depth_mm", hue="sex",
                        style="sex", markers=markers, palette=colors, s=55, ax=ax)
        sns.regplot(data=subset, x="bill_length_mm", y="bill_depth_mm", scatter=False,
                    color="#526171", ci=None, ax=ax)
        ax.set_title(species_name)
        ax.set(xlabel="Bill length (mm)", ylabel="Bill depth (mm)")
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(title or "Penguin bill measurements by species and sex")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig
'''

FLIGHTS_LINES = '''def plot_data(data, *, title=None, figsize=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    frame = data.copy()
    fig, ax = plt.subplots(figsize=figsize or (11, 5.5))
    sns.lineplot(data=frame, x="month", y="passengers", hue="year", palette="viridis",
                 legend=False, ax=ax)
    ax.set(xlabel="Month", ylabel="Passengers")
    ax.set_title(title or "Monthly airline passengers by year")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig
'''

FLIGHTS_HEATMAP = '''def plot_data(data, *, title=None, figsize=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    frame = data.copy()
    frame["month"] = frame["month"].astype(str).str[:3]
    matrix = frame.pivot(index="year", columns="month", values="passengers").reindex(
        columns=month_order
    )
    fig, ax = plt.subplots(figsize=figsize or (12, 7))
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="mako", cbar_kws={"label": "Passengers"},
                ax=ax)
    peak = frame.loc[frame["passengers"].idxmax()]
    peak_col = month_order.index(peak["month"])
    peak_row = list(matrix.index).index(peak["year"])
    ax.add_patch(plt.Rectangle((peak_col, peak_row), 1, 1, fill=False,
                               edgecolor="#ED8500", linewidth=3))
    ax.set(xlabel="Month", ylabel="Year")
    ax.set_title(title or "Monthly airline passengers | peak outlined")
    fig.tight_layout()
    return fig
'''

FLIGHTS_PLOTLY = '''def plot_data(data, *, title=None, figsize=None):
    import plotly.graph_objects as go
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    frame = data.copy()
    frame["month"] = frame["month"].astype(str).str[:3]
    matrix = frame.pivot(index="year", columns="month", values="passengers").reindex(
        columns=month_order
    )
    fig = go.Figure(go.Heatmap(x=matrix.columns, y=matrix.index, z=matrix.to_numpy(),
                               colorscale="Viridis", colorbar={"title": "Passengers"},
                               hovertemplate="%{x} %{y}<br>%{z} passengers<extra></extra>"))
    fig.update_layout(title=title or "Monthly airline passengers", xaxis_title="Month",
                      yaxis_title="Year")
    if figsize is not None:
        fig.update_layout(width=figsize[0] * 100, height=figsize[1] * 100)
    return fig
'''
