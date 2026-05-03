#!/usr/bin/env python3
"""
gen_scatter.py  --  generate a training-data scatter plot as an SVG file.

Usage:
    python gen_scatter.py light
    python gen_scatter.py dark
"""

import sys
import math
import random
import svgwrite

# ---------------------------------------------------------------------------
# Model consts
# ---------------------------------------------------------------------------
SLOPE = 0.6
Y_INTERCEPT = 5.2
N_POINTS = 150
X_MIN, X_MAX = -5.0, 15.0
Y_MIN, Y_MAX = -1.0, 16.0

# ---------------------------------------------------------------------------
# Jitter consts
#   JITTER_SIGMA     -- std-dev of the "normal" noise band, as a fraction of
#                       the y data range
#   OUTLIER_FRACTION -- share of points that are outliers
#   OUTLIER_SCALE    -- how far outliers can reach (fraction of y data range)
# ---------------------------------------------------------------------------
Y_RANGE = Y_MAX - Y_MIN
JITTER_SIGMA = 0.07  # most points within ~1 sigma = 7% of y range
OUTLIER_FRACTION = 0.08  # ~8% of points are outliers
OUTLIER_SCALE = 2.50  # outliers up to 250% of y range away

# ---------------------------------------------------------------------------
# Layout consts
# ---------------------------------------------------------------------------
IMG_HEIGHT = 250  # px, total SVG height
PADDING = 30  # px, around the plot area
TABLE_WIDTH = 90  # px, for the x/y table on the right
GAP = 20  # px, between plot area and table

# Derived
PLOT_HEIGHT = IMG_HEIGHT - 2 * PADDING

# Table rows: figure out how many (x,y) pairs fit given row height
TABLE_ROW_H = 16  # px per data row
TABLE_HEADER_H = TABLE_ROW_H
TABLE_DOT_H = TABLE_ROW_H
AVAILABLE_H = PLOT_HEIGHT - TABLE_HEADER_H - 2 * TABLE_DOT_H
TABLE_N_ROWS = max(1, math.floor(AVAILABLE_H / TABLE_ROW_H))

PLOT_WIDTH = 340  # px, width of just the plot area
IMG_WIDTH = PADDING + PLOT_WIDTH + GAP + TABLE_WIDTH + PADDING

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUT_DIR = "book/images/backprop"
OUT_BASENAME = "xy-plot"

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
THEMES = {
    "light": {
        "bg": "white",
        "axis": "#111111",
        "grid": "#cccccc",
        "tick": "#111111",
        "dot": "#1a5fa8",
        "dot_alpha": 0.75,
        "table_text": "#111111",
        "table_dim": "#888888",
        "table_line": "#dddddd",
    },
    "dark": {
        "bg": "#1e1e1e",
        "axis": "#eeeeee",
        "grid": "#444444",
        "tick": "#eeeeee",
        "dot": "#5b9bd5",
        "dot_alpha": 0.80,
        "table_text": "#dddddd",
        "table_dim": "#777777",
        "table_line": "#444444",
    },
}


# ---------------------------------------------------------------------------
# Jitter function
# ---------------------------------------------------------------------------
JITTER_POWER = 4.5    # higher = more points near the line, wilder outliers
JITTER_MAX   = 2.5    # max jitter as a fraction of Y_RANGE
JITTER_EXTRA = 0.4    # extra wiggle in data units (not a fraction)

def jitter(y_ideal: float) -> float:
    t     = random.random()
    scale = (t ** JITTER_POWER) * JITTER_MAX
    sign  = random.choice([-1, 1])
    extra = random.gauss(0, JITTER_EXTRA)
    return y_ideal + sign * scale * Y_RANGE + extra


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------
def data_to_svg(x: float, y: float) -> tuple[float, float]:
    """Map data coordinates to SVG pixel coordinates within the plot area."""
    px = PADDING + (x - X_MIN) / (X_MAX - X_MIN) * PLOT_WIDTH
    py = PADDING + (Y_MAX - y) / (Y_MAX - Y_MIN) * PLOT_HEIGHT
    return px, py


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("light", "dark"):
        print("Usage: python gen_scatter.py [light|dark]")
        sys.exit(1)

    style = sys.argv[1]
    c = THEMES[style]
    path = f"{OUT_DIR}/{OUT_BASENAME}-{style}.svg"

    random.seed(42)  # reproducible output

    # --- generate points ----------------------------------------------------
    points = []
    for _ in range(N_POINTS):
        x = random.uniform(X_MIN, X_MAX)
        y_ideal = SLOPE * x + Y_INTERCEPT
        y = jitter(y_ideal)
        points.append((round(x, 2), round(y, 2)))

    # --- SVG canvas ---------------------------------------------------------
    dwg = svgwrite.Drawing(
        path,
        size=(f"{IMG_WIDTH}px", f"{IMG_HEIGHT}px"),
        profile="full",
    )

    # background
    dwg.add(dwg.rect(insert=(0, 0), size=(IMG_WIDTH, IMG_HEIGHT), fill=c["bg"]))

    # --- grid (light lines at each integer tick) ----------------------------
    x_ticks = range(math.ceil(X_MIN), math.floor(X_MAX) + 1)
    y_ticks = range(math.ceil(Y_MIN), math.floor(Y_MAX) + 1)

    for xi in x_ticks:
        px, _ = data_to_svg(xi, 0)
        _, py_top = data_to_svg(0, Y_MAX)
        _, py_bottom = data_to_svg(0, Y_MIN)
        dwg.add(
            dwg.line(
                start=(px, py_top),
                end=(px, py_bottom),
                stroke=c["grid"],
                stroke_width=0.5,
            )
        )

    for yi in y_ticks:
        _, py = data_to_svg(0, yi)
        px_left, _ = data_to_svg(X_MIN, 0)
        px_right, _ = data_to_svg(X_MAX, 0)
        dwg.add(
            dwg.line(
                start=(px_left, py),
                end=(px_right, py),
                stroke=c["grid"],
                stroke_width=0.5,
            )
        )

    # --- axes (x=0 and y=0) as bold lines -----------------------------------
    ax0, _ = data_to_svg(0, 0)
    _, ay_top = data_to_svg(0, Y_MAX)
    _, ay_bottom = data_to_svg(0, Y_MIN)
    _, ay0 = data_to_svg(0, 0)
    ax_left, _ = data_to_svg(X_MIN, 0)
    ax_right, _ = data_to_svg(X_MAX, 0)

    # y-axis (x=0 vertical line)
    dwg.add(
        dwg.line(
            start=(ax0, ay_top),
            end=(ax0, ay_bottom),
            stroke=c["axis"],
            stroke_width=2.5,
        )
    )

    # x-axis (y=0 horizontal line)
    dwg.add(
        dwg.line(
            start=(ax_left, ay0),
            end=(ax_right, ay0),
            stroke=c["axis"],
            stroke_width=2.5,
        )
    )

    # --- tick marks (small nubs on the axes, no labels) ---------------------
    TICK_LEN = 4

    for xi in x_ticks:
        if xi == 0:
            continue
        px, _ = data_to_svg(xi, 0)
        dwg.add(
            dwg.line(
                start=(px, ay0 - TICK_LEN),
                end=(px, ay0 + TICK_LEN),
                stroke=c["tick"],
                stroke_width=1,
            )
        )

    for yi in y_ticks:
        if yi == 0:
            continue
        _, py = data_to_svg(0, yi)
        dwg.add(
            dwg.line(
                start=(ax0 - TICK_LEN, py),
                end=(ax0 + TICK_LEN, py),
                stroke=c["tick"],
                stroke_width=1,
            )
        )

    # --- data points --------------------------------------------------------
    for x, y in points:
        # skip points outside the plot area
        if not (X_MIN <= x <= X_MAX and Y_MIN <= y <= Y_MAX):
            continue
        px, py = data_to_svg(x, y)
        dwg.add(
            dwg.circle(
                center=(px, py),
                r=1.5,
                fill=c["dot"],
                opacity=c["dot_alpha"],
            )
        )

    # --- table on the right -------------------------------------------------
    tx = PADDING + PLOT_WIDTH + GAP  # left edge of table
    ty = PADDING  # top of table

    col_x_cx = tx + 22  # x-column text centre
    col_y_cx = tx + 68  # y-column text centre

    # header labels
    for text, cx in [("x", col_x_cx), ("y", col_y_cx)]:
        dwg.add(
            dwg.text(
                text,
                insert=(cx, ty + 11),
                text_anchor="middle",
                font_family="monospace",
                font_size="11px",
                fill=c["table_dim"],
            )
        )

    # header separator line
    dwg.add(
        dwg.line(
            start=(tx, ty + TABLE_HEADER_H + 2),
            end=(tx + TABLE_WIDTH - 4, ty + TABLE_HEADER_H + 2),
            stroke=c["table_line"],
            stroke_width=0.5,
        )
    )

    mid_cx = (col_x_cx + col_y_cx) // 2


    ellipsis = '...'

    # data rows
    row_y = ty + TABLE_HEADER_H + TABLE_DOT_H
    for i, (x, y) in enumerate(points[:TABLE_N_ROWS]):
        baseline = row_y + i * TABLE_ROW_H
        dwg.add(
            dwg.text(
                f"{x:.1f}",
                insert=(col_x_cx, baseline),
                text_anchor="middle",
                font_family="monospace",
                font_size="11px",
                fill=c["table_text"],
            )
        )
        dwg.add(
            dwg.text(
                f"{y:.1f}",
                insert=(col_y_cx, baseline),
                text_anchor="middle",
                font_family="monospace",
                font_size="11px",
                fill=c["table_text"],
            )
        )

    # bottom "…"
    bottom_y = row_y + TABLE_N_ROWS * TABLE_ROW_H + 8
    dwg.add(
        dwg.text(
            ellipsis,
            insert=(mid_cx, bottom_y),
            text_anchor="middle",
            font_family="monospace",
            font_size="11px",
            fill=c["table_dim"],
        )
    )

    # --- save ---------------------------------------------------------------
    import os

    os.makedirs(OUT_DIR, exist_ok=True)
    dwg.save()
    print(f"Written: {path}")


if __name__ == "__main__":
    main()
