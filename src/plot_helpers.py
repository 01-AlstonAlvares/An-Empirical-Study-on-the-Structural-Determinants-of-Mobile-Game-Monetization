"""Standardized plot styling and figure export utilities for publication-grade visuals."""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from .config import FIGURES_DIR
except ImportError:
    from src.config import FIGURES_DIR


def set_plot_style():
    """Apply consistent, professional styling across all notebooks."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 15,
            "figure.autolayout": True,
            "axes.edgecolor": "#cccccc",
            "axes.linewidth": 0.8,
            "grid.color": "#e5e5e5",
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
        }
    )


def save_figure(fig: plt.Figure, filename: str) -> Path:
    """Save matplotlib figure to artifacts/figures directory with standardized resolution."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return out_path
