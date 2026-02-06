**Version**: 1.0

# Plotting Style Guide — `regimes`

All plots produced by the `regimes` package must follow this style guide.
The visual identity draws from The Economist, the Financial Times, and Edward Tufte's
principles: maximise data-ink, minimise chart junk, and let the data speak.

---

## Design Rules

| Rule | Implementation |
|------|----------------|
| **Spines** | Only left + bottom. Never top or right. Left spine thin (`0.8`). |
| **Gridlines** | Horizontal only (`axes.grid.axis: 'y'`). Light grey, thin, behind data. |
| **Fonts** | Sans-serif throughout: Helvetica → Arial → DejaVu Sans (fallback chain). |
| **Font sizes** | Title 13pt bold, axis labels 11pt, tick labels 10pt, annotations 9–10pt. |
| **Title** | States the insight or describes what is shown. Left-aligned with the y-axis. |
| **Subtitle** | Optional. Below the title in regular weight, slightly smaller or grey. |
| **Source line** | Small muted text at bottom-left if applicable. |
| **Legends** | Avoid. Use direct line labeling at the endpoint of each series. Fall back to a legend only when there are 4+ overlapping series. Legends must be frameless. |
| **Colors** | Use the Economist-inspired palette below. 2–3 colors per plot max. Grey for context/secondary data, color for emphasis. |
| **Confidence bands** | Translucent `fill_between` (alpha 0.15–0.25), same hue as the line, no edge. |
| **Break dates** | Thin dashed vertical lines in muted grey. Optionally annotated at the top. |
| **Regime shading** | Alternating very light tints (barely visible), never saturated. |
| **Tick marks** | No tick marks (size 0). Padding only. |
| **Axis labels** | Concise. Y-axis label horizontal below the title, or as a subtitle — not rotated 90° if it can be avoided. |
| **Figure size** | Default `(10, 5)`. Aspect ratio roughly 2:1 for time series. |
| **DPI** | 150 for screen, 300 for saved files. |

---

## Color Palette

Primary Economist-inspired cycle, in order of preference:

| Name | Hex | Usage |
|------|-----|-------|
| Blue | `#006BA2` | Primary series, parameter estimates |
| Red | `#DB444B` | Secondary series, rejection regions, significance |
| Teal | `#3EBCD2` | Tertiary series |
| Green | `#379A8B` | Alternative series |
| Gold | `#EBB434` | Highlights, warnings |
| Grey | `#758D99` | Context lines, break dates, secondary elements |
| Mauve | `#9A607F` | Additional series if needed |

Supplementary:

| Name | Hex | Usage |
|------|-----|-------|
| Light grey | `#d4d4d4` | Gridlines |
| Near-black | `#333333` | Text, axis lines |
| Regime tint A | `#f0f4f8` | Light blue-grey for odd regimes |
| Regime tint B | `#ffffff` | White for even regimes |
| CI band | Same as line color | At alpha 0.15–0.20 |

---

## Default Style Configuration

The package should provide a function that sets `rcParams` for all plots. This function
must be called at the start of every public plotting method.

```python
import matplotlib.pyplot as plt
import matplotlib as mpl

REGIMES_COLORS = [
    '#006BA2', '#DB444B', '#3EBCD2', '#379A8B',
    '#EBB434', '#758D99', '#9A607F',
]

REGIMES_STYLE = {
    # Figure
    'figure.figsize': (10, 5),
    'figure.dpi': 150,
    'figure.facecolor': 'white',

    # Font
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 11,

    # Axes
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.linewidth': 0.8,
    'axes.edgecolor': '#333333',
    'axes.labelsize': 11,
    'axes.labelcolor': '#333333',
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.titlepad': 15,
    'axes.grid': True,
    'axes.grid.axis': 'y',
    'axes.axisbelow': True,
    'axes.prop_cycle': plt.cycler('color', REGIMES_COLORS),

    # Grid
    'grid.color': '#d4d4d4',
    'grid.linewidth': 0.5,
    'grid.alpha': 0.7,

    # Lines
    'lines.linewidth': 2.0,
    'lines.antialiased': True,

    # Ticks
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'xtick.major.size': 0,
    'ytick.major.size': 0,
    'xtick.major.pad': 8,
    'ytick.major.pad': 8,

    # Legend
    'legend.frameon': False,
    'legend.fontsize': 10,

    # Saving
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2,
}


def set_style():
    """Apply the regimes plotting style globally."""
    mpl.rcParams.update(REGIMES_STYLE)


def use_style():
    """Context manager for temporary style application."""
    return plt.style.context(REGIMES_STYLE)
```

---

## Plotting Helper Functions

These reusable helpers should live in the plotting module and be used by all plot functions.

### Direct Line Labeling

```python
def label_line_end(ax, x, y, label, color, offset=(8, 0), fontsize=10):
    """Label a line at its last data point (replaces legend)."""
    ax.annotate(
        label,
        xy=(x[-1], y[-1]),
        xytext=offset,
        textcoords='offset points',
        fontsize=fontsize,
        color=color,
        va='center',
        fontweight='bold',
        clip_on=False,
    )
```

### Break Date Markers

```python
def add_break_dates(ax, break_dates, color='#758D99', linestyle='--',
                    linewidth=0.8, alpha=0.7, labels=None):
    """Add vertical dashed lines at structural break dates."""
    for i, bd in enumerate(break_dates):
        ax.axvline(x=bd, color=color, linestyle=linestyle,
                   linewidth=linewidth, alpha=alpha, zorder=1)
        if labels and i < len(labels):
            ax.text(bd, ax.get_ylim()[1], f' {labels[i]}',
                    fontsize=9, color=color, alpha=alpha,
                    va='top', ha='left', rotation=0)
```

### Confidence Bands

```python
def add_confidence_band(ax, x, lower, upper, color='#006BA2', alpha=0.15):
    """Add a translucent confidence interval band."""
    ax.fill_between(x, lower, upper, color=color, alpha=alpha,
                    linewidth=0, zorder=0)
```

### Regime Shading

```python
def shade_regimes(ax, break_dates, start, end,
                  colors=('#f0f4f8', '#ffffff'), alpha=0.5):
    """Shade alternating regimes with barely-visible tints."""
    boundaries = [start] + list(break_dates) + [end]
    for i in range(len(boundaries) - 1):
        ax.axvspan(boundaries[i], boundaries[i + 1],
                   facecolor=colors[i % len(colors)], alpha=alpha, zorder=0)
```

### Source Annotation

```python
def add_source(ax, text, fontsize=8, color='#999999'):
    """Add a source/note line at the bottom-left of the plot."""
    ax.text(0, -0.12, text, transform=ax.transAxes,
            fontsize=fontsize, color=color, va='top', ha='left')
```

---

## Plot Types and Their Design

Each public plot method must call `set_style()` (or use `use_style()` as a context
manager) before creating the figure. All plots must accept an optional `ax` argument
so users can embed them in their own figures.

### Parameter Estimates with Structural Breaks

The primary plot type. Shows how regression coefficients evolve across regimes.

- **Horizontal line segments** for each regime's parameter estimate (step function), colored with the primary blue.
- **Confidence bands** around each segment (translucent fill).
- **Break dates** as thin dashed vertical lines in grey.
- **Regime shading** as alternating light tints (optional, off by default).
- **Direct labels** for each coefficient if multiple parameters are shown.
- **Title example**: "Coefficient estimates with 2 structural breaks"
- **No legend** if ≤3 series; direct labeling instead.

### CUSUM / Test Statistic Plots

- **Test statistic** as a solid line (primary blue).
- **Critical value boundaries** as horizontal dashed lines (red `#DB444B`).
- **Rejection region** shaded lightly in red (alpha 0.08–0.12).
- **Detected break points** marked with a vertical line + small annotation.
- **Title example**: "CUSUM test statistic"

### Break Date Confidence Intervals

- **Horizontal dumbbell/lollipop chart**: one row per break.
- **Point estimate** as a filled circle (primary blue).
- **Confidence interval** as a horizontal bar (same blue, thinner).
- **Clean, minimal** — no gridlines needed for this chart type.
- **Y-axis labels**: "Break 1", "Break 2", etc.

### Residual Plots

- **Scatter** of residuals over time (small dots, alpha 0.5–0.7).
- **Zero line** as a solid thin line (near-black).
- **Break dates** as vertical dashed lines.
- **Regime shading** optional.

### Fitted vs Actual

- **Actual data** as a thin grey line or small dots.
- **Fitted values** as a solid colored line (primary blue), thicker.
- **Break dates** as vertical dashed lines.
- **Direct labeling**: "Actual" (grey) and "Fitted" (blue) at endpoints.

### F-statistic / Information Criterion Plots

- **Line or bar chart** of test statistics across candidate break numbers.
- **Highlight the selected model** (e.g., the bar for the chosen number of breaks in a different color).
- **Critical value** as a horizontal dashed line if applicable.

---

## Implementation Checklist

When implementing or refactoring a plot function, verify:

- [ ] Calls `set_style()` or uses `use_style()` context manager
- [ ] Accepts optional `ax` parameter (creates `fig, ax` only if `ax is None`)
- [ ] Returns `(fig, ax)` tuple
- [ ] Uses only colors from `REGIMES_COLORS` or the supplementary palette
- [ ] No top or right spines
- [ ] Only horizontal gridlines
- [ ] Direct line labels instead of legend (when ≤3 series)
- [ ] Frameless legend (when legend is necessary)
- [ ] Title is descriptive
- [ ] Confidence bands use `fill_between` with low alpha
- [ ] Break dates use thin dashed grey verticals
- [ ] Sans-serif font throughout
- [ ] Saved output at 300 DPI

---

## Reference Sources

Style decisions in this guide draw from:

- The Economist visual style guide and chart conventions
- The Financial Times Visual Vocabulary
- Edward Tufte, *The Visual Display of Quantitative Information*
- Robert Ritz, "Making Economist-Style Plots in Matplotlib"
- Python Graph Gallery curated best examples
- Nicolas Rougier, *Scientific Visualization: Python + Matplotlib*
- Coding for Economists, "Narrative Data Visualisation" chapter