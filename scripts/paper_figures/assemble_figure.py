"""Assemble a multi-panel figure from panel images (PNG) into one PDF (+ PNG preview).

A layout is a list of rows; each row is a list of (letter, image_path) pairs. Within a row the panels share one
height and keep their aspect ratios; rows are stacked and scaled to a common width. Panel letters are drawn in regular weight
at the top-left of each panel. Images are embedded at their native resolution.

usage (from Python):
    from assemble_figure import assemble
    assemble([[("A", "a.png"), ("B", "b.png")]], "Fig3", width_in=7.0)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "notebooks", "output", "paper_figures", "figures"))
Image.MAX_IMAGE_PIXELS = None


def assemble(rows, name, width_in=7.0, gap=0.02, letter_size=18, row_labels=None, row_label_width=0.0, letter_pad=0.03):
    """rows: list of lists of (letter or None, path). gap: fraction of the figure width between panels.
    row_labels: optional list of strings drawn vertically at the left of each row (e.g. tool names)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    imgs = [[(letter, Image.open(p).convert("RGBA"), p) for letter, p in row] for row in rows]
    usable = 1.0 - row_label_width
    # per row: height (in figure-width units) so the row's panels plus gaps span the usable width
    row_heights = []
    for row in imgs:
        aspect_sum = sum(im.width / im.height for _, im, _ in row)
        pads = sum(letter_pad for letter, _, _ in row if letter)
        row_heights.append((usable - gap * (len(row) - 1) - pads) / aspect_sum)
    total_h = sum(row_heights) + gap * (len(rows) - 1)
    fig = plt.figure(figsize=(width_in, width_in * total_h))
    y = total_h
    for r, (row, h) in enumerate(zip(imgs, row_heights)):
        y -= h
        x = row_label_width
        if row_labels:
            fig.text(row_label_width * 0.45, (y + h / 2) / total_h, row_labels[r], rotation=90, ha="center",
                     va="center", fontsize=letter_size - 2)
        for letter, im, _ in row:
            if letter:
                fig.text(x, (y + h) / total_h, letter, fontsize=letter_size, fontweight="normal", ha="left", va="top")
                x += letter_pad
            w = h * im.width / im.height
            ax = fig.add_axes([x, y / total_h, w, h / total_h])
            ax.imshow(im, interpolation="none")
            ax.set_axis_off()
            x += w + gap
        y -= gap
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"wrote {OUT_DIR}/{name}.pdf ({width_in:.1f} x {width_in * total_h:.1f} in)")


def combine(paths, titles, out_png, title_size=14, gap_px=40, title_x=0.5):
    """Place images side by side (bottom-aligned) with a title above each; returns out_png.
    title_x: horizontal position of each title as a fraction of its image width (e.g. to centre it over the plot
    area rather than plot + legend). Used for multi-image panels such as the raw/processed dot plots."""
    ims = [Image.open(p).convert("RGBA") for p in paths]
    h = max(im.height for im in ims)
    w = sum(im.width for im in ims) + gap_px * (len(ims) - 1)
    dpi = 300
    title_h = int(title_size / 72 * dpi * 1.6)
    fig = plt.figure(figsize=(w / dpi, (h + title_h) / dpi), dpi=dpi)
    x = 0
    for im, t in zip(ims, titles):
        ax = fig.add_axes([x / w, 0, im.width / w, im.height / (h + title_h)])
        ax.imshow(im, interpolation="none"); ax.set_axis_off()
        fig.text((x + title_x * im.width) / w, 1 - 0.1 * title_h / (h + title_h), t, ha="center", va="top", fontsize=title_size)
        x += im.width + gap_px
    fig.savefig(out_png, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_png


def titled(path, lines, out_png, title_size=26, bold=True):
    """Stack a multi-line centred title above an image; returns out_png (sizes in points at the image's pixel scale)."""
    im = Image.open(path).convert("RGBA")
    dpi = 300
    line_h = int(title_size / 72 * dpi * 1.25)
    th = line_h * len(lines) + line_h // 2
    fig = plt.figure(figsize=(im.width / dpi, (im.height + th) / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, im.height / (im.height + th)]); ax.imshow(im, interpolation="none"); ax.set_axis_off()
    fig.text(0.5, 1 - (line_h // 4) / (im.height + th), "\n".join(lines), ha="center", va="top", fontsize=title_size,
             fontweight="bold" if bold else "normal", linespacing=1.25)
    fig.savefig(out_png, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_png


def side_label(path, text, out_png, label_size=40, label_frac=0.13):
    """Add a vertical label to the left of an image (for per-panel tool names); returns out_png."""
    im = Image.open(path).convert("RGBA")
    dpi = 300
    lw = int(im.width * label_frac)
    fig = plt.figure(figsize=((im.width + lw) / dpi, im.height / dpi), dpi=dpi)
    ax = fig.add_axes([lw / (im.width + lw), 0, im.width / (im.width + lw), 1]); ax.imshow(im, interpolation="none"); ax.set_axis_off()
    fig.text(0.45 * lw / (im.width + lw), 0.5, text, rotation=90, ha="center", va="center", fontsize=label_size)
    fig.savefig(out_png, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_png


def pad_width(path, width_frac, out_png):
    """Centre an image on a white canvas so it occupies width_frac of the canvas width (for a narrower full-row panel)."""
    im = Image.open(path).convert("RGBA")
    W = int(im.width / width_frac)
    canvas = Image.new("RGBA", (W, im.height), (255, 255, 255, 255))
    canvas.alpha_composite(im, ((W - im.width) // 2, 0))
    canvas.save(out_png)
    return out_png
