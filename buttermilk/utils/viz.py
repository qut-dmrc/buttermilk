"""Visualization defaults and helpers for Buttermilk.

Provides opinionated, high-quality defaults for matplotlib, seaborn, and plotly
with profiles optimized for different contexts (HiDPI displays, print, web).

Quick start:
    >>> from buttermilk.utils.viz import init_viz
    >>> init_viz()  # Auto-detects best profile
    >>> # Or specify:
    >>> init_viz(profile="hidpi")  # For 4K/5K displays
    >>> init_viz(profile="print")  # For publication-quality output

Color themes:
    - cyberpunk (default): Neon colors on dark background, highly readable
    - academic: Colorblind-safe palette for publications
    - minimal: Clean, minimal design
"""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

try:
    import plotly.graph_objects as go
    import plotly.io as pio

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


ProfileType = Literal["hidpi", "print", "notebook", "web"]
ThemeType = Literal["cyberpunk", "academic", "minimal"]


# === COLOR PALETTES ===

PALETTES = {
    "cyberpunk": {
        "primary": "#00FFF0",  # Bright cyan
        "secondary": "#FF006E",  # Hot pink
        "accent": "#FFBE0B",  # Electric yellow
        "success": "#00F5A0",  # Neon green
        "warning": "#FF5E00",  # Bright orange
        "error": "#FF0054",  # Neon red
        "background": "#0A0E27",  # Deep blue-black
        "surface": "#1A1F3A",  # Slightly lighter
        "text": "#E0E7FF",  # Soft white
        "grid": "#2D3561",  # Subtle grid
        # Categorical palette (8 colors, highly distinguishable)
        "categorical": [
            "#00FFF0",  # Cyan
            "#FF006E",  # Pink
            "#FFBE0B",  # Yellow
            "#00F5A0",  # Green
            "#9D4EDD",  # Purple
            "#FF5E00",  # Orange
            "#06FFA5",  # Mint
            "#FF0054",  # Red
        ],
        # Sequential palette (light to dark)
        "sequential": ["#0A0E27", "#1A3A52", "#2D6A6A", "#4A9B83", "#7FD1AE", "#B8FFE5"],
    },
    "academic": {
        "primary": "#0173B2",  # Blue (colorblind safe)
        "secondary": "#DE8F05",  # Orange
        "accent": "#029E73",  # Teal
        "success": "#029E73",  # Teal
        "warning": "#DE8F05",  # Orange
        "error": "#CC78BC",  # Purple
        "background": "#FFFFFF",  # White
        "surface": "#F8F9FA",  # Light gray
        "text": "#2E3440",  # Dark gray
        "grid": "#ECEFF4",  # Very light gray
        # Colorblind-safe categorical (Okabe-Ito palette)
        "categorical": [
            "#0173B2",  # Blue
            "#DE8F05",  # Orange
            "#029E73",  # Teal
            "#CC78BC",  # Purple
            "#CA9161",  # Tan
            "#FBAFE4",  # Pink
            "#949494",  # Gray
            "#ECE133",  # Yellow
        ],
        "sequential": ["#F7FBFF", "#DEEBF7", "#C6DBEF", "#9ECAE1", "#6BAED6", "#3182BD"],
    },
    "minimal": {
        "primary": "#2E3440",  # Dark gray
        "secondary": "#5E81AC",  # Muted blue
        "accent": "#88C0D0",  # Light blue
        "success": "#A3BE8C",  # Green
        "warning": "#EBCB8B",  # Yellow
        "error": "#BF616A",  # Red
        "background": "#FFFFFF",  # White
        "surface": "#ECEFF4",  # Light gray
        "text": "#2E3440",  # Dark gray
        "grid": "#E5E9F0",  # Very light gray
        "categorical": ["#5E81AC", "#BF616A", "#A3BE8C", "#EBCB8B", "#B48EAD", "#88C0D0"],
        "sequential": ["#ECEFF4", "#E5E9F0", "#D8DEE9", "#88C0D0", "#5E81AC", "#4C566A"],
    },
}


# === PROFILE CONFIGURATIONS ===

PROFILES = {
    "hidpi": {
        "dpi": 300,
        "figsize": (16, 10),  # Larger for 5K display
        "font_size": 14,
        "title_size": 18,
        "label_size": 14,
        "tick_size": 12,
        "legend_size": 12,
        "linewidth": 2.5,
        "markersize": 8,
        "context": "talk",  # Seaborn context
    },
    "print": {
        "dpi": 600,  # High quality for print
        "figsize": (8, 6),  # Standard print size
        "font_size": 10,
        "title_size": 12,
        "label_size": 10,
        "tick_size": 9,
        "legend_size": 9,
        "linewidth": 1.5,
        "markersize": 5,
        "context": "paper",
    },
    "notebook": {
        "dpi": 150,
        "figsize": (10, 6),
        "font_size": 12,
        "title_size": 14,
        "label_size": 12,
        "tick_size": 10,
        "legend_size": 10,
        "linewidth": 2,
        "markersize": 6,
        "context": "notebook",
    },
    "web": {
        "dpi": 150,
        "figsize": (12, 7),
        "font_size": 13,
        "title_size": 16,
        "label_size": 13,
        "tick_size": 11,
        "legend_size": 11,
        "linewidth": 2.5,
        "markersize": 7,
        "context": "notebook",
    },
}


# === INITIALIZATION ===


def init_viz(
    profile: ProfileType = "hidpi",
    theme: ThemeType = "cyberpunk",
    set_style: bool = True,
    set_plotly: bool = True,
) -> dict:
    """Initialize visualization defaults with one call.

    Args:
        profile: Display profile - hidpi (5K), print, notebook, or web
        theme: Color theme - cyberpunk (default), academic, or minimal
        set_style: Whether to apply matplotlib/seaborn styles
        set_plotly: Whether to configure plotly (if available)

    Returns:
        dict: Configuration applied (profile + palette)

    Example:
        >>> from buttermilk.utils.viz import init_viz
        >>> config = init_viz()  # HiDPI cyberpunk theme
        >>> config = init_viz(profile="print", theme="academic")  # Publication
    """
    profile_config = PROFILES[profile]
    palette = PALETTES[theme]

    if set_style:
        _apply_matplotlib_style(profile_config, palette)
        _apply_seaborn_style(profile_config, palette, theme)

    if set_plotly and PLOTLY_AVAILABLE:
        _apply_plotly_style(profile_config, palette, theme)

    return {"profile": profile_config, "palette": palette, "theme": theme}


def _apply_matplotlib_style(profile: dict, palette: dict) -> None:
    """Apply matplotlib rcParams."""
    plt.rcParams.update(
        {
            # Figure
            "figure.dpi": profile["dpi"],
            "figure.figsize": profile["figsize"],
            "figure.facecolor": palette["background"],
            "figure.edgecolor": palette["background"],
            # Axes
            "axes.facecolor": palette["background"],
            "axes.edgecolor": palette["grid"],
            "axes.labelcolor": palette["text"],
            "axes.labelsize": profile["label_size"],
            "axes.titlesize": profile["title_size"],
            "axes.titleweight": "bold",
            "axes.linewidth": 1.5,
            "axes.grid": True,
            "axes.axisbelow": True,  # Grid behind data
            # Grid
            "grid.color": palette["grid"],
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.3,
            # Ticks
            "xtick.labelsize": profile["tick_size"],
            "ytick.labelsize": profile["tick_size"],
            "xtick.color": palette["text"],
            "ytick.color": palette["text"],
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            # Lines
            "lines.linewidth": profile["linewidth"],
            "lines.markersize": profile["markersize"],
            "lines.markeredgewidth": 0,
            # Legend
            "legend.fontsize": profile["legend_size"],
            "legend.frameon": True,
            "legend.framealpha": 0.8,
            "legend.facecolor": palette["surface"],
            "legend.edgecolor": palette["grid"],
            # Text
            "text.color": palette["text"],
            "font.size": profile["font_size"],
            "font.family": "sans-serif",
            # Saving
            "savefig.dpi": profile["dpi"],
            "savefig.facecolor": palette["background"],
            "savefig.edgecolor": palette["background"],
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )


def _apply_seaborn_style(profile: dict, palette: dict, theme: str) -> None:
    """Apply seaborn styling."""
    # Set context (paper, notebook, talk, poster)
    sns.set_context(profile["context"])

    # Set color palette
    sns.set_palette(palette["categorical"])

    # Set style
    if theme == "cyberpunk":
        sns.set_style(
            "darkgrid",
            {
                "axes.facecolor": palette["background"],
                "figure.facecolor": palette["background"],
                "grid.color": palette["grid"],
                "text.color": palette["text"],
                "axes.labelcolor": palette["text"],
                "xtick.color": palette["text"],
                "ytick.color": palette["text"],
            },
        )
    elif theme == "academic":
        sns.set_style(
            "whitegrid",
            {
                "axes.facecolor": palette["background"],
                "grid.color": palette["grid"],
            },
        )
    else:  # minimal
        sns.set_style(
            "ticks",
            {
                "axes.facecolor": palette["background"],
                "grid.color": palette["grid"],
            },
        )


def _apply_plotly_style(profile: dict, palette: dict, theme: str) -> None:
    """Apply plotly template."""
    if not PLOTLY_AVAILABLE:
        return

    # Create custom template
    template = go.layout.Template()

    # Layout defaults
    template.layout = go.Layout(
        font=dict(family="sans-serif", size=profile["font_size"], color=palette["text"]),
        plot_bgcolor=palette["background"],
        paper_bgcolor=palette["background"],
        title=dict(font=dict(size=profile["title_size"], color=palette["text"])),
        xaxis=dict(
            gridcolor=palette["grid"],
            linecolor=palette["grid"],
            tickfont=dict(size=profile["tick_size"], color=palette["text"]),
            titlefont=dict(size=profile["label_size"], color=palette["text"]),
            showgrid=True,
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor=palette["grid"],
            linecolor=palette["grid"],
            tickfont=dict(size=profile["tick_size"], color=palette["text"]),
            titlefont=dict(size=profile["label_size"], color=palette["text"]),
            showgrid=True,
            zeroline=False,
        ),
        legend=dict(
            font=dict(size=profile["legend_size"], color=palette["text"]),
            bgcolor=palette["surface"],
            bordercolor=palette["grid"],
            borderwidth=1,
        ),
        colorway=palette["categorical"],
    )

    # Register template
    pio.templates[f"buttermilk_{theme}"] = template
    pio.templates.default = f"buttermilk_{theme}"


# === HELPER FUNCTIONS ===


def get_palette(theme: ThemeType = "cyberpunk") -> dict:
    """Get color palette for a theme.

    Returns:
        dict with keys: primary, secondary, accent, categorical, sequential, etc.
    """
    return PALETTES[theme].copy()


def get_categorical_colors(theme: ThemeType = "cyberpunk", n: int | None = None) -> list[str]:
    """Get categorical color list.

    Args:
        theme: Theme name
        n: Number of colors (cycles if more than available)

    Returns:
        List of hex color strings
    """
    colors = PALETTES[theme]["categorical"]
    if n is None:
        return colors
    if n <= len(colors):
        return colors[:n]
    # Cycle colors if more needed
    return [colors[i % len(colors)] for i in range(n)]


def get_sequential_colors(theme: ThemeType = "cyberpunk", n: int = 6, reverse: bool = False) -> list[str]:
    """Get sequential color gradient.

    Args:
        theme: Theme name
        n: Number of color steps
        reverse: Reverse the gradient (dark to light)

    Returns:
        List of hex color strings
    """
    colors = PALETTES[theme]["sequential"]
    if reverse:
        colors = colors[::-1]

    # Interpolate if needed
    if n == len(colors):
        return colors

    # Simple interpolation for now
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n)
    return [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b, _ in cmap(np.linspace(0, 1, n))]


def quick_figure(
    nrows: int = 1,
    ncols: int = 1,
    theme: ThemeType | None = None,
    profile: ProfileType | None = None,
    **kwargs,
) -> tuple:
    """Create figure with current style.

    Args:
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        theme: Override theme (uses current if None)
        profile: Override profile (uses current if None)
        **kwargs: Passed to plt.subplots()

    Returns:
        (fig, axes) tuple

    Example:
        >>> fig, ax = quick_figure()
        >>> ax.plot([1, 2, 3], [1, 4, 9])
        >>> plt.show()
    """
    if theme or profile:
        # Reinitialize with new settings
        init_viz(profile=profile or "hidpi", theme=theme or "cyberpunk")

    return plt.subplots(nrows, ncols, **kwargs)


def save_figure(
    fig,
    filename: str,
    dpi: int | None = None,
    formats: list[str] | None = None,
    transparent: bool = False,
) -> list[str]:
    """Save figure in multiple formats.

    Args:
        fig: Matplotlib figure
        filename: Base filename (without extension)
        dpi: Override DPI (uses profile default if None)
        formats: List of formats ['png', 'svg', 'pdf'] (default: ['png'])
        transparent: Transparent background

    Returns:
        List of saved file paths

    Example:
        >>> fig, ax = quick_figure()
        >>> ax.plot([1, 2, 3])
        >>> paths = save_figure(fig, "my_plot", formats=["png", "svg"])
    """
    from pathlib import Path

    if formats is None:
        formats = ["png"]

    if dpi is None:
        dpi = plt.rcParams["savefig.dpi"]

    saved_paths = []
    base_path = Path(filename).with_suffix("")

    for fmt in formats:
        save_path = f"{base_path}.{fmt}"
        fig.savefig(save_path, dpi=dpi, transparent=transparent, bbox_inches="tight")
        saved_paths.append(save_path)

    return saved_paths


# === STYLE CONTEXT MANAGERS ===


class temp_style:
    """Temporarily apply a different style.

    Example:
        >>> with temp_style(profile="print", theme="academic"):
        ...     fig, ax = plt.subplots()
        ...     ax.plot([1, 2, 3])
        ...     plt.savefig("paper_figure.png")
        >>> # Original style restored
    """

    def __init__(self, profile: ProfileType | None = None, theme: ThemeType | None = None):
        self.profile = profile
        self.theme = theme
        self.old_rc = None

    def __enter__(self):
        # Save current state
        self.old_rc = plt.rcParams.copy()
        # Apply new style
        if self.profile or self.theme:
            init_viz(profile=self.profile or "hidpi", theme=self.theme or "cyberpunk")
        return self

    def __exit__(self, *args):
        # Restore old state
        plt.rcParams.update(self.old_rc)


# === CONVENIENCE FUNCTIONS ===


def cyberpunk_glow(ax, color: str | None = None, intensity: float = 0.5) -> None:  # type: ignore[no-untyped-def]
    """Add neon glow effect to plot lines (cyberpunk theme).

    Args:
        ax: Matplotlib axes
        color: Glow color (uses line color if None)
        intensity: Glow intensity (0-1)

    Example:
        >>> fig, ax = quick_figure()
        >>> ax.plot([1, 2, 3], [1, 4, 9], color="#00FFF0")
        >>> cyberpunk_glow(ax)
    """
    for line in ax.get_lines():
        glow_color = color or line.get_color()
        # Add multiple offset shadows for glow effect
        for width in [8, 6, 4, 2]:
            ax.plot(
                line.get_xdata(),
                line.get_ydata(),
                color=glow_color,
                linewidth=line.get_linewidth() + width,
                alpha=intensity * 0.15,
                zorder=line.get_zorder() - 1,
            )
