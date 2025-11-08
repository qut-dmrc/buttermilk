"""Examples of using buttermilk.utils.viz for beautiful plots.

Run this file to generate example plots in multiple themes and profiles.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from buttermilk.utils.viz import (
    cyberpunk_glow,
    get_categorical_colors,
    get_palette,
    init_viz,
    quick_figure,
    save_figure,
    temp_style,
)


def example_line_plot():
    """Example: Line plot with cyberpunk theme."""
    # Initialize with HiDPI profile (your 5K display)
    init_viz(profile="hidpi", theme="cyberpunk")

    # Create figure
    fig, ax = quick_figure()

    # Generate data
    x = np.linspace(0, 10, 100)
    palette = get_palette("cyberpunk")

    # Plot multiple lines
    ax.plot(x, np.sin(x), label="sin(x)", color=palette["primary"], linewidth=3)
    ax.plot(x, np.cos(x), label="cos(x)", color=palette["secondary"], linewidth=3)
    ax.plot(
        x,
        np.sin(x) * np.cos(x),
        label="sin(x)·cos(x)",
        color=palette["accent"],
        linewidth=3,
    )

    # Add glow effect for cyberpunk aesthetic
    cyberpunk_glow(ax, intensity=0.6)

    # Labels
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Cyberpunk Waveforms")
    ax.legend()

    # Save in multiple formats
    save_figure(fig, "example_cyberpunk_lines", formats=["png", "svg"])
    plt.close()


def example_scatter_plot():
    """Example: Scatter plot with academic theme."""
    init_viz(profile="print", theme="academic")

    fig, ax = quick_figure()

    # Generate random data
    np.random.seed(42)
    n = 200
    x = np.random.randn(n)
    y = 2 * x + np.random.randn(n) * 0.5
    colors = get_categorical_colors("academic", n=3)

    # Create categories
    categories = np.random.choice(["Group A", "Group B", "Group C"], n)

    for i, (cat, color) in enumerate(zip(["Group A", "Group B", "Group C"], colors)):
        mask = categories == cat
        ax.scatter(
            x[mask], y[mask], label=cat, color=color, s=50, alpha=0.7, edgecolors="none"
        )

    ax.set_xlabel("Variable X")
    ax.set_ylabel("Variable Y")
    ax.set_title("Academic Publication Style")
    ax.legend()

    save_figure(fig, "example_academic_scatter", formats=["png", "pdf"])
    plt.close()


def example_heatmap():
    """Example: Heatmap with minimal theme."""
    init_viz(profile="web", theme="minimal")

    # Generate correlation matrix
    np.random.seed(42)
    data = np.random.randn(10, 12)
    corr = np.corrcoef(data)

    fig, ax = quick_figure(figsize=(12, 10))

    # Create heatmap
    im = ax.imshow(corr, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation", rotation=270, labelpad=20)

    # Labels
    ax.set_title("Correlation Heatmap - Minimal Style")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels([f"Var{i}" for i in range(10)])
    ax.set_yticklabels([f"Var{i}" for i in range(10)])

    save_figure(fig, "example_minimal_heatmap", formats=["png"])
    plt.close()


def example_multi_theme():
    """Example: Same plot in different themes."""
    # Generate data
    x = np.linspace(0, 2 * np.pi, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    themes = ["cyberpunk", "academic", "minimal"]

    for theme in themes:
        with temp_style(profile="notebook", theme=theme):
            fig, ax = quick_figure()

            palette = get_palette(theme)
            ax.plot(x, y1, label="sin(x)", color=palette["primary"], linewidth=2.5)
            ax.plot(x, y2, label="cos(x)", color=palette["secondary"], linewidth=2.5)

            if theme == "cyberpunk":
                cyberpunk_glow(ax, intensity=0.5)

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"{theme.title()} Theme")
            ax.legend()

            save_figure(fig, f"example_theme_{theme}", formats=["png"])
            plt.close()


def example_seaborn_integration():
    """Example: Seaborn plots with custom themes."""
    init_viz(profile="hidpi", theme="cyberpunk")

    # Load example dataset
    tips = sns.load_dataset("tips")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Violin plot
    sns.violinplot(data=tips, x="day", y="total_bill", ax=axes[0, 0])
    axes[0, 0].set_title("Violin Plot")

    # Box plot
    sns.boxplot(data=tips, x="day", y="tip", ax=axes[0, 1])
    axes[0, 1].set_title("Box Plot")

    # Scatter plot
    sns.scatterplot(
        data=tips, x="total_bill", y="tip", hue="time", ax=axes[1, 0], s=100, alpha=0.7
    )
    axes[1, 0].set_title("Scatter Plot")

    # Bar plot
    sns.barplot(data=tips, x="day", y="total_bill", estimator=np.mean, ax=axes[1, 1])
    axes[1, 1].set_title("Bar Plot")

    plt.tight_layout()
    save_figure(fig, "example_seaborn_cyberpunk", formats=["png"])
    plt.close()


def example_print_vs_screen():
    """Example: Same plot optimized for print vs screen."""
    x = np.linspace(0, 10, 100)
    y = np.exp(-x / 5) * np.sin(2 * x)

    # Screen version (HiDPI)
    init_viz(profile="hidpi", theme="cyberpunk")
    fig, ax = quick_figure()
    palette = get_palette("cyberpunk")
    ax.plot(x, y, color=palette["primary"], linewidth=3)
    cyberpunk_glow(ax)
    ax.set_title("Screen Optimized (HiDPI)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Signal")
    save_figure(fig, "example_screen", formats=["png"])
    plt.close()

    # Print version
    init_viz(profile="print", theme="academic")
    fig, ax = quick_figure()
    palette = get_palette("academic")
    ax.plot(x, y, color=palette["primary"], linewidth=2)
    ax.set_title("Print Optimized (600 DPI)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Signal")
    save_figure(fig, "example_print", formats=["pdf", "png"])
    plt.close()


def run_all_examples():
    """Run all example functions."""
    # Create output directory
    output_dir = Path("viz_examples")
    output_dir.mkdir(exist_ok=True)

    import os

    os.chdir(output_dir)

    print("Generating visualization examples...")
    print("1/7: Line plot (cyberpunk theme)...")
    example_line_plot()

    print("2/7: Scatter plot (academic theme)...")
    example_scatter_plot()

    print("3/7: Heatmap (minimal theme)...")
    example_heatmap()

    print("4/7: Multi-theme comparison...")
    example_multi_theme()

    print("5/7: Seaborn integration...")
    example_seaborn_integration()

    print("6/7: Print vs screen optimization...")
    example_print_vs_screen()

    print(f"\n✨ All examples generated in {output_dir.absolute()}/")
    print("\nQuick reference:")
    print("  from buttermilk.utils.viz import init_viz, quick_figure")
    print("  init_viz()  # One-line setup")
    print("  fig, ax = quick_figure()  # Ready to plot")


if __name__ == "__main__":
    run_all_examples()
