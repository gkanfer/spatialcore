import pandas as pd

from spatialcore.plotting import plot_dotplot, plot_spatial_scatter


def test_plot_spatial_scatter_returns_fig_ax():
    df = pd.DataFrame({"x": [0, 1], "y": [0, 1], "value": [1.0, 2.0]})
    fig, ax = plot_spatial_scatter(df, color="value", show=False)
    assert fig is ax.figure


def test_plot_dotplot_returns_fig_ax():
    df = pd.DataFrame({"x": ["a", "b"], "y": ["p1", "p2"], "size": [1, 2], "score": [0.1, 0.2]})
    fig, ax = plot_dotplot(df, x="x", y="y", size="size", color="score", show=False)
    assert fig is ax.figure
