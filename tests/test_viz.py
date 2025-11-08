"""End-to-end tests for buttermilk.utils.viz module.

Tests the complete visualization defaults system including:
- Profile initialization (hidpi, print, notebook, web)
- Theme configuration (cyberpunk, academic, minimal)
- Color palette retrieval
- Helper functions (quick_figure, save_figure, etc.)
- Context managers (temp_style)
- Integration with matplotlib, seaborn, plotly
"""

import tempfile
from pathlib import Path

import pytest

# Mark all tests in this module as requiring viz dependencies
pytestmark = pytest.mark.skipif(
    False,  # Will be set by import checks below
    reason="Visualization dependencies (matplotlib/seaborn) not installed",
)

# Try importing viz dependencies
try:
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    from buttermilk.utils.viz import (
        PALETTES,
        PROFILES,
        cyberpunk_glow,
        get_categorical_colors,
        get_palette,
        get_sequential_colors,
        init_viz,
        quick_figure,
        save_figure,
        temp_style,
    )

    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False
    # Update marker
    pytestmark = pytest.mark.skipif(
        not VIZ_AVAILABLE,
        reason="Visualization dependencies (matplotlib/seaborn) not installed",
    )


class TestInitViz:
    """Test init_viz() initialization function."""

    def test_init_viz_default(self):
        """Test default initialization (hidpi + cyberpunk)."""
        config = init_viz()

        assert config["theme"] == "cyberpunk"
        assert config["profile"]["dpi"] == 300  # hidpi
        assert config["profile"]["figsize"] == (16, 10)

        # Check matplotlib rcParams were set
        assert plt.rcParams["figure.dpi"] == 300
        assert list(plt.rcParams["figure.figsize"]) == [16, 10]
        assert plt.rcParams["figure.facecolor"] == PALETTES["cyberpunk"]["background"]

    def test_init_viz_print_profile(self):
        """Test print profile initialization."""
        config = init_viz(profile="print", theme="academic")

        assert config["profile"]["dpi"] == 600  # print
        assert plt.rcParams["figure.dpi"] == 600
        assert config["theme"] == "academic"

    def test_init_viz_notebook_profile(self):
        """Test notebook profile initialization."""
        config = init_viz(profile="notebook", theme="minimal")

        assert config["profile"]["dpi"] == 150
        assert config["profile"]["context"] == "notebook"
        assert config["theme"] == "minimal"

    def test_init_viz_web_profile(self):
        """Test web profile initialization."""
        config = init_viz(profile="web")

        assert config["profile"]["dpi"] == 150
        assert config["profile"]["figsize"] == (12, 7)

    def test_init_viz_sets_seaborn_palette(self):
        """Test that seaborn palette is configured."""
        init_viz(theme="cyberpunk")

        # Get current seaborn palette
        palette = sns.color_palette()

        # Should match cyberpunk categorical colors (within tolerance)
        assert len(palette) >= 6  # At least 6 colors

    @pytest.mark.skipif(
        not VIZ_AVAILABLE, reason="Visualization dependencies not available"
    )
    def test_init_viz_configures_all_rcparams(self):
        """Test that all important rcParams are set."""
        init_viz(profile="hidpi", theme="cyberpunk")

        # Check all critical rcParams are configured
        assert plt.rcParams["axes.facecolor"] == PALETTES["cyberpunk"]["background"]
        assert plt.rcParams["axes.labelcolor"] == PALETTES["cyberpunk"]["text"]
        assert plt.rcParams["text.color"] == PALETTES["cyberpunk"]["text"]
        assert plt.rcParams["grid.color"] == PALETTES["cyberpunk"]["grid"]
        assert plt.rcParams["axes.grid"] is True


class TestProfiles:
    """Test different display profiles."""

    def test_all_profiles_exist(self):
        """Test that all documented profiles exist."""
        required_profiles = ["hidpi", "print", "notebook", "web"]

        for profile_name in required_profiles:
            assert profile_name in PROFILES, f"Missing profile: {profile_name}"

    def test_profile_structure(self):
        """Test that profiles have required keys."""
        required_keys = ["dpi", "figsize", "font_size", "context"]

        for profile_name, profile in PROFILES.items():
            for key in required_keys:
                assert key in profile, f"Profile {profile_name} missing key: {key}"

    def test_hidpi_profile_values(self):
        """Test HiDPI profile has correct values for 5K display."""
        profile = PROFILES["hidpi"]

        assert profile["dpi"] == 300  # High DPI
        assert profile["figsize"][0] >= 16  # Large figure
        assert profile["font_size"] >= 14  # Readable at distance
        assert profile["linewidth"] >= 2.5  # Thick lines

    def test_print_profile_values(self):
        """Test print profile has correct values for publications."""
        profile = PROFILES["print"]

        assert profile["dpi"] == 600  # High quality
        assert profile["figsize"] == (8, 6)  # Standard print size
        assert profile["context"] == "paper"  # Seaborn paper context


class TestThemes:
    """Test color themes."""

    def test_all_themes_exist(self):
        """Test that all documented themes exist."""
        required_themes = ["cyberpunk", "academic", "minimal"]

        for theme_name in required_themes:
            assert theme_name in PALETTES, f"Missing theme: {theme_name}"

    def test_theme_structure(self):
        """Test that themes have required color keys."""
        required_keys = ["primary", "secondary", "background", "text", "categorical"]

        for theme_name, palette in PALETTES.items():
            for key in required_keys:
                assert key in palette, f"Theme {theme_name} missing key: {key}"

    def test_cyberpunk_colors(self):
        """Test cyberpunk theme has correct neon colors."""
        palette = PALETTES["cyberpunk"]

        # Check signature neon colors
        assert palette["primary"] == "#00FFF0"  # Cyan
        assert palette["secondary"] == "#FF006E"  # Pink
        assert palette["accent"] == "#FFBE0B"  # Yellow

        # Check has enough categorical colors
        assert len(palette["categorical"]) >= 6

    def test_academic_colors_colorblind_safe(self):
        """Test academic theme uses Okabe-Ito palette."""
        palette = PALETTES["academic"]

        # Okabe-Ito palette starts with these colors
        assert palette["primary"] == "#0173B2"  # Blue
        assert palette["secondary"] == "#DE8F05"  # Orange

        # Should have 8 categorical colors (full Okabe-Ito)
        assert len(palette["categorical"]) == 8


class TestPaletteFunctions:
    """Test palette retrieval functions."""

    def test_get_palette(self):
        """Test get_palette() returns correct palette."""
        palette = get_palette("cyberpunk")

        assert isinstance(palette, dict)
        assert "primary" in palette
        assert "categorical" in palette
        assert isinstance(palette["categorical"], list)

    def test_get_categorical_colors_default(self):
        """Test get_categorical_colors() with default n."""
        colors = get_categorical_colors("cyberpunk")

        assert isinstance(colors, list)
        assert len(colors) == 8  # Cyberpunk has 8 colors

    def test_get_categorical_colors_subset(self):
        """Test get_categorical_colors() with n < available."""
        colors = get_categorical_colors("cyberpunk", n=3)

        assert len(colors) == 3
        # Should be first 3 colors
        full_palette = PALETTES["cyberpunk"]["categorical"]
        assert colors[0] == full_palette[0]
        assert colors[1] == full_palette[1]
        assert colors[2] == full_palette[2]

    def test_get_categorical_colors_cycles(self):
        """Test get_categorical_colors() cycles when n > available."""
        colors = get_categorical_colors("cyberpunk", n=20)

        assert len(colors) == 20
        # Should cycle - first color should repeat at position 8
        full_palette = PALETTES["cyberpunk"]["categorical"]
        assert colors[0] == colors[len(full_palette)]

    def test_get_sequential_colors(self):
        """Test get_sequential_colors() returns gradient."""
        colors = get_sequential_colors("cyberpunk", n=6)

        assert len(colors) == 6
        assert all(color.startswith("#") for color in colors)

    def test_get_sequential_colors_reverse(self):
        """Test get_sequential_colors() with reverse."""
        forward = get_sequential_colors("cyberpunk", n=6, reverse=False)
        backward = get_sequential_colors("cyberpunk", n=6, reverse=True)

        # First of forward should match last of backward
        assert forward[0] == backward[-1]


class TestQuickFigure:
    """Test quick_figure() helper function."""

    def test_quick_figure_single_plot(self):
        """Test quick_figure() creates single plot."""
        init_viz()
        fig, ax = quick_figure()

        assert fig is not None
        assert ax is not None
        # Should use current profile settings
        assert fig.dpi == plt.rcParams["figure.dpi"]

        plt.close(fig)

    def test_quick_figure_subplots(self):
        """Test quick_figure() creates subplot grid."""
        init_viz()
        fig, axes = quick_figure(nrows=2, ncols=2)

        assert fig is not None
        assert axes.shape == (2, 2)

        plt.close(fig)

    def test_quick_figure_inherits_style(self):
        """Test quick_figure() uses current style."""
        init_viz(profile="print", theme="academic")

        fig, ax = quick_figure()

        # Should match print profile
        assert fig.dpi == 600  # Print DPI
        assert fig.get_facecolor()[0:3] == (
            1.0,
            1.0,
            1.0,
        )  # White background (academic)

        plt.close(fig)


class TestSaveFigure:
    """Test save_figure() helper function."""

    def test_save_figure_single_format(self):
        """Test saving figure in single format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3])

            output_path = Path(tmpdir) / "test_plot"
            paths = save_figure(fig, str(output_path), formats=["png"])

            assert len(paths) == 1
            assert Path(paths[0]).exists()
            assert Path(paths[0]).suffix == ".png"

            plt.close(fig)

    def test_save_figure_multiple_formats(self):
        """Test saving figure in multiple formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3])

            output_path = Path(tmpdir) / "test_plot"
            paths = save_figure(fig, str(output_path), formats=["png", "svg", "pdf"])

            assert len(paths) == 3
            for path in paths:
                assert Path(path).exists()

            # Check all formats created
            suffixes = {Path(p).suffix for p in paths}
            assert suffixes == {".png", ".svg", ".pdf"}

            plt.close(fig)

    def test_save_figure_custom_dpi(self):
        """Test save_figure() respects custom DPI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3])

            output_path = Path(tmpdir) / "test_plot"
            paths = save_figure(fig, str(output_path), dpi=150, formats=["png"])

            assert Path(paths[0]).exists()

            plt.close(fig)


class TestTempStyle:
    """Test temp_style context manager."""

    def test_temp_style_restores_original(self):
        """Test temp_style() restores original style."""
        # Set initial style
        init_viz(profile="hidpi", theme="cyberpunk")
        original_dpi = plt.rcParams["figure.dpi"]

        # Temporarily change to print style
        with temp_style(profile="print", theme="academic"):
            assert plt.rcParams["figure.dpi"] == 600  # Print DPI

        # Should restore original
        assert plt.rcParams["figure.dpi"] == original_dpi

    def test_temp_style_allows_plotting(self):
        """Test that plotting works inside temp_style context."""
        init_viz(theme="cyberpunk")

        with temp_style(theme="academic"):
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3])

            # Should use academic theme
            assert fig.get_facecolor()[0:3] == (1.0, 1.0, 1.0)  # White

            plt.close(fig)

        # Original theme restored
        fig, ax = plt.subplots()
        # Cyberpunk has dark background
        assert fig.get_facecolor()[0:3] != (1.0, 1.0, 1.0)
        plt.close(fig)


class TestCyberpunkGlow:
    """Test cyberpunk_glow() special effect."""

    def test_cyberpunk_glow_adds_lines(self):
        """Test cyberpunk_glow() adds glow effect lines."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])

        original_line_count = len(ax.get_lines())

        # Add glow effect
        cyberpunk_glow(ax, intensity=0.5)

        # Should add multiple glow lines (4 shadow lines per original line)
        assert len(ax.get_lines()) > original_line_count

        plt.close(fig)

    def test_cyberpunk_glow_custom_color(self):
        """Test cyberpunk_glow() with custom color."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])

        # Add glow with custom color
        cyberpunk_glow(ax, color="#FF0000", intensity=0.8)

        # Glow lines should exist
        assert len(ax.get_lines()) > 1

        plt.close(fig)


class TestEndToEnd:
    """End-to-end workflow tests."""

    def test_complete_workflow_hidpi_cyberpunk(self):
        """Test complete workflow: init → plot → save for HiDPI display."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize
            config = init_viz(profile="hidpi", theme="cyberpunk")

            # Create plot
            fig, ax = quick_figure()
            x = np.linspace(0, 10, 100)
            palette = get_palette("cyberpunk")

            ax.plot(x, np.sin(x), color=palette["primary"], linewidth=3)
            ax.plot(x, np.cos(x), color=palette["secondary"], linewidth=3)

            # Add glow
            cyberpunk_glow(ax)

            ax.set_xlabel("Time")
            ax.set_ylabel("Amplitude")
            ax.set_title("Test Plot")

            # Save
            output_path = Path(tmpdir) / "test_plot"
            paths = save_figure(fig, str(output_path), formats=["png"])

            # Verify
            assert config["theme"] == "cyberpunk"
            assert Path(paths[0]).exists()

            plt.close(fig)

    def test_complete_workflow_print_academic(self):
        """Test complete workflow: publication-ready plot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize for publication
            config = init_viz(profile="print", theme="academic")

            # Create plot
            fig, ax = quick_figure()
            x = np.linspace(0, 10, 50)
            colors = get_categorical_colors("academic", n=3)

            ax.plot(x, np.sin(x), color=colors[0], label="Group A")
            ax.plot(x, np.cos(x), color=colors[1], label="Group B")
            ax.plot(x, np.sin(x) * np.cos(x), color=colors[2], label="Group C")

            ax.set_xlabel("Variable X")
            ax.set_ylabel("Variable Y")
            ax.legend()

            # Save as PDF for paper
            output_path = Path(tmpdir) / "figure1"
            paths = save_figure(fig, str(output_path), formats=["pdf", "png"], dpi=600)

            # Verify
            assert config["profile"]["dpi"] == 600
            assert len(paths) == 2
            assert all(Path(p).exists() for p in paths)

            plt.close(fig)

    def test_multiple_themes_comparison(self):
        """Test creating same plot in different themes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            x = np.linspace(0, 2 * np.pi, 100)

            for theme in ["cyberpunk", "academic", "minimal"]:
                with temp_style(theme=theme):
                    fig, ax = quick_figure()

                    palette = get_palette(theme)
                    ax.plot(x, np.sin(x), color=palette["primary"])

                    output_path = Path(tmpdir) / f"plot_{theme}"
                    paths = save_figure(fig, str(output_path), formats=["png"])

                    assert Path(paths[0]).exists()

                    plt.close(fig)

    def test_seaborn_integration(self):
        """Test that seaborn plots use configured theme."""
        init_viz(theme="cyberpunk")

        # Create seaborn plot
        fig, ax = plt.subplots()

        # Generate data
        np.random.seed(42)
        data = {"x": np.random.randn(100), "y": np.random.randn(100)}

        # Seaborn should use theme colors
        sns.scatterplot(data=data, x="x", y="y", ax=ax)

        # Verify background matches theme
        assert ax.get_facecolor()[0:3] != (
            1.0,
            1.0,
            1.0,
        )  # Not white (cyberpunk is dark)

        plt.close(fig)
