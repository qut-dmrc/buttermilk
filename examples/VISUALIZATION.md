# Buttermilk Visualization System

Beautiful, opinionated defaults for matplotlib, seaborn, and plotly with profiles optimized for different contexts.

## Quick Start

```python
from buttermilk.utils.viz import init_viz

# One line - done! Uses HiDPI cyberpunk theme by default
init_viz()

# Now just plot as normal
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
plt.show()
```

## Profiles

Optimized for different display/output contexts:

- **`hidpi`** (default): For 4K/5K displays (your 40" 5K setup!)

  - 300 DPI, large fonts, thick lines
  - Optimized for high-res monitors

- **`print`**: Publication-quality output

  - 600 DPI, smaller text, fine lines
  - Ready for academic journals

- **`notebook`**: Jupyter notebook display

  - 150 DPI, medium sizing
  - Balanced for inline display

- **`web`**: Web embedding

  - 150 DPI, slightly larger
  - Good for dashboards/web apps

## Themes

### Cyberpunk (Default)

Neon colors on dark background. High contrast, highly readable, looks amazing on your 5K display.

```python
init_viz(profile="hidpi", theme="cyberpunk")
```

**Colors:**

- Primary: Bright cyan (#00FFF0)
- Secondary: Hot pink (#FF006E)
- Accent: Electric yellow (#FFBE0B)
- Success: Neon green (#00F5A0)
- Background: Deep blue-black (#0A0E27)

**Special effect:**

```python
from buttermilk.utils.viz import cyberpunk_glow

# Add neon glow to your plots
ax.plot(x, y)
cyberpunk_glow(ax, intensity=0.6)
```

### Academic

Colorblind-safe Okabe-Ito palette. Perfect for publications.

```python
init_viz(profile="print", theme="academic")
```

**Colors:** Blue, orange, teal, purple (all colorblind-friendly)

### Minimal

Clean, Scandinavian-inspired design. Subtle and professional.

```python
init_viz(profile="web", theme="minimal")
```

## Helper Functions

### Quick Figure Creation

```python
from buttermilk.utils.viz import quick_figure

# Creates figure with current style
fig, ax = quick_figure()
ax.plot([1, 2, 3])
```

### Save in Multiple Formats

```python
from buttermilk.utils.viz import save_figure

fig, ax = plt.subplots()
ax.plot([1, 2, 3])

# Save as PNG, SVG, and PDF in one call
paths = save_figure(fig, "my_plot", formats=["png", "svg", "pdf"])
```

### Get Color Palettes

```python
from buttermilk.utils.viz import get_palette, get_categorical_colors

# Get full palette dict
palette = get_palette("cyberpunk")
ax.plot(x, y, color=palette["primary"])

# Get N categorical colors
colors = get_categorical_colors("cyberpunk", n=5)
for i, color in enumerate(colors):
    ax.plot(x, data[i], color=color)
```

### Temporary Style Changes

```python
from buttermilk.utils.viz import temp_style

# Use different style temporarily
with temp_style(profile="print", theme="academic"):
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    plt.savefig("paper_figure.pdf")

# Original style restored automatically
```

## Common Workflows

### Notebook Analysis

```python
from buttermilk.utils import nb_init  # Imports viz automatically!

# This sets up everything including cyberpunk viz theme
bm = nb_init(job="my_analysis", project="my_project")

# Ready to plot!
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(data)
```

### Publication Figures

```python
from buttermilk.utils.viz import init_viz, save_figure

# High-res, colorblind-safe academic theme
init_viz(profile="print", theme="academic")

fig, ax = plt.subplots()
ax.plot(x, y1, label="Control")
ax.plot(x, y2, label="Treatment")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Response")
ax.legend()

# Save as PDF (vector) for paper + PNG for presentations
save_figure(fig, "figure1_results", formats=["pdf", "png"], dpi=600)
```

### Screen Presentation

```python
from buttermilk.utils.viz import init_viz, cyberpunk_glow

# Optimized for your 5K display
init_viz(profile="hidpi", theme="cyberpunk")

fig, ax = plt.subplots(figsize=(16, 10))  # Large for 5K
ax.plot(x, y, linewidth=3, color="#00FFF0")
cyberpunk_glow(ax, intensity=0.7)  # Extra dramatic for presentations
plt.show()
```

### Multiple Plots, Different Themes

```python
from buttermilk.utils.viz import init_viz

datasets = load_data()

for theme in ["cyberpunk", "academic", "minimal"]:
    init_viz(theme=theme)

    fig, ax = plt.subplots()
    ax.plot(datasets["x"], datasets["y"])
    ax.set_title(f"{theme.title()} Theme")

    plt.savefig(f"plot_{theme}.png")
    plt.close()
```

## Seaborn Integration

All seaborn plots automatically use the theme:

```python
import seaborn as sns
from buttermilk.utils.viz import init_viz

init_viz(theme="cyberpunk")

# Seaborn uses theme colors automatically
sns.scatterplot(data=df, x="x", y="y", hue="category")
sns.violinplot(data=df, x="group", y="value")
```

## Plotly Integration

Plotly templates are automatically configured:

```python
import plotly.graph_objects as go
from buttermilk.utils.viz import init_viz

init_viz(theme="cyberpunk")  # Sets plotly template

# Plotly uses theme automatically
fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[1, 4, 9]))
fig.show()
```

## Examples

Run the examples script to see all themes and profiles:

```bash
python examples/visualization_examples.py
```

This generates example plots in `viz_examples/` directory showing:

- All three themes
- Different plot types
- Seaborn integration
- Print vs screen optimization
- Special effects (cyberpunk glow)

## Tips

### For Your 5K Display

```python
# Perfect for your setup
init_viz(profile="hidpi", theme="cyberpunk")

# Use larger figure sizes
fig, ax = plt.subplots(figsize=(16, 12))
```

### For Papers

```python
# Publication ready
init_viz(profile="print", theme="academic")

# Always save as vector (PDF/SVG)
save_figure(fig, "figure", formats=["pdf", "png"])
```

### For Dashboards

```python
# Web optimized
init_viz(profile="web", theme="minimal")
```

## Color Reference

### Cyberpunk Palette

| Color           | Hex       | Usage                       |
| --------------- | --------- | --------------------------- |
| Bright Cyan     | `#00FFF0` | Primary lines, main data    |
| Hot Pink        | `#FF006E` | Secondary lines, comparison |
| Electric Yellow | `#FFBE0B` | Accents, highlights         |
| Neon Green      | `#00F5A0` | Success, positive trends    |
| Bright Orange   | `#FF5E00` | Warnings, attention         |
| Neon Red        | `#FF0054` | Errors, negative trends     |

### Academic Palette (Okabe-Ito)

Scientifically proven colorblind-safe colors.

| Color  | Hex       | Usage      |
| ------ | --------- | ---------- |
| Blue   | `#0173B2` | Primary    |
| Orange | `#DE8F05` | Secondary  |
| Teal   | `#029E73` | Tertiary   |
| Purple | `#CC78BC` | Quaternary |

All 8 colors are distinguishable by people with all types of color vision deficiency.

## Philosophy

**Opinionated defaults that look great out of the box.**

- One line to set up (`init_viz()`)
- Beautiful by default (cyberpunk theme)
- Optimized for your hardware (HiDPI profile)
- Publication-ready when needed (print profile)
- Accessible (academic theme is colorblind-safe)
- No fiddling with rcParams - it just works
