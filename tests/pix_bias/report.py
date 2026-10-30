"""Compare 10 uA vs 20 uA pixel bias: horizontal profiles left and right of the saturated spot.

  python -m tests.pix_bias.report [--corrected output/pix_bias]

Reads the *_dc.tif files written by tests.pix_bias.correct and writes report.html and
profiles.csv next to them. A profile is the mean of a band of rows through the spot,
plotted against distance from the spot edge, overlaying the two bias currents.
"""
import argparse
import base64
import csv
import io
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.patches import Rectangle
from scipy import ndimage

BIASES = (10, 20)
COLORS = {10: "tab:blue", 20: "tab:orange"}
NEAR, FAR_WIDTH = (10, 60), 50  # px from spot edge: "near" zone, width of "far" baseline zone


def find_spot(img):
    """Largest saturated blob (median filter removes hot pixels): centroid row, x extent."""
    smooth = ndimage.median_filter(img, size=5)
    labels, n = ndimage.label(smooth > 0.95 * smooth.max())
    biggest = 1 + int(np.argmax(ndimage.sum(np.ones_like(labels), labels, range(1, n + 1))))
    ys, xs = np.nonzero(labels == biggest)
    return {"cy": int(round(ys.mean())), "x0": int(xs.min()), "x1": int(xs.max()), "r": int(round(np.sqrt(len(xs) / np.pi)))}


def profiles(img, spot, half, span):
    band = img[spot["cy"] - half:spot["cy"] + half + 1].mean(0)
    d = np.arange(1, span + 1)
    return d, {"left": band[spot["x0"] - d], "right": band[spot["x1"] + d]}


def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def stats(d, prof):
    near = prof[(d >= NEAR[0]) & (d <= NEAR[1])].mean()
    far = prof[d > d.max() - FAR_WIDTH].mean()
    return {"far_adu": far, "near_adu": near, "excess_adu": near - far}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--corrected", type=Path, default=Path("output/pix_bias"))
    p.add_argument("--half-band", type=int, default=10, help="rows above/below the spot centre to average")
    p.add_argument("--span", type=int, default=250, help="profile length (px) each side of the spot")
    args = p.parse_args()

    files = {}
    for f in args.corrected.glob("*_dc.tif"):
        m = re.match(r"(\d+)ms_pbc_(\d+)uA_dc\.tif", f.name)
        if m and int(m.group(2)) in BIASES:
            files[(int(m.group(1)), int(m.group(2)))] = f
    exps = sorted({e for e, _ in files if all((e, b) in files for b in BIASES)})
    ref_img = tifffile.imread(files[(exps[-1], BIASES[0])])
    spot = find_spot(ref_img)
    print("spot:", spot)

    rows, figs = [], []
    for e in exps:
        data = {b: profiles(tifffile.imread(files[(e, b)]), spot, args.half_band, args.span) for b in BIASES}
        d = data[BIASES[0]][0]
        fig, ax = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
        for col, side in enumerate(("left", "right")):
            for b in BIASES:
                prof = data[b][1][side]
                ax[0, col].plot(d, prof, color=COLORS[b], lw=0.8, label=f"{b} uA")
                rows.append({"exposure_ms": e, "bias_uA": b, "side": side, **stats(d, prof)})
            ax[0, col].set_yscale("log")
            ax[0, col].set_title(f"{e} ms, {side} of spot")
            ax[0, col].set_ylabel("ADU (dark-corrected)")
            ax[0, col].legend()
            far = d >= NEAR[0]  # the first pixels are the spot's own edge, not bloom
            ax[1, col].plot(d[far], (data[20][1][side] - data[10][1][side])[far], color="k", lw=0.8)
            ax[1, col].axhline(0, c="gray", lw=0.5)
            ax[1, col].set_ylabel("20 uA - 10 uA (ADU)")
            ax[1, col].set_xlabel("distance from spot edge (px)")
        figs.append((e, png(fig)))

    with open(args.corrected / "profiles.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # overview image with band and profile windows
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(ref_img, cmap="gray", vmin=np.percentile(ref_img, 1), vmax=np.percentile(ref_img, 99.5))
    for x in (spot["x0"] - args.span, spot["x1"]):
        ax.add_patch(Rectangle((x, spot["cy"] - args.half_band), args.span, 2 * args.half_band + 1,
                               fill=False, ec="red", lw=1.5))
    ax.set_title(f"{exps[-1]} ms, {BIASES[0]} uA: profile bands")
    overview = png(fig)

    # summary: near-spot excess vs exposure
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for col, side in enumerate(("left", "right")):
        for b in BIASES:
            rs = [r for r in rows if r["side"] == side and r["bias_uA"] == b]
            ax[col].plot([r["exposure_ms"] for r in rs], [r["excess_adu"] for r in rs], "o-", color=COLORS[b], label=f"{b} uA")
        ax[col].set_xscale("log"); ax[col].set_title(f"{side}: near-spot excess"); ax[col].set_xlabel("exposure (ms)")
        ax[col].axhline(0, c="gray", lw=0.5); ax[col].legend()
    ax[0].set_ylabel(f"mean({NEAR[0]}-{NEAR[1]} px) - far baseline (ADU)")
    summary = png(fig)

    table = ["<table border=1 cellpadding=4 style='border-collapse:collapse'><tr><th>exposure</th><th>side</th>"
             + "".join(f"<th>{b} uA excess (ADU)</th>" for b in BIASES) + "<th>20/10 excess</th></tr>"]
    for e in exps:
        for side in ("left", "right"):
            r = {b: next(x for x in rows if (x["exposure_ms"], x["bias_uA"], x["side"]) == (e, b, side)) for b in BIASES}
            ratio = r[20]["excess_adu"] / r[10]["excess_adu"] if r[10]["excess_adu"] else float("nan")
            table.append(f"<tr><td>{e} ms</td><td>{side}</td>"
                         + "".join(f"<td>{r[b]['excess_adu']:.1f}</td>" for b in BIASES)
                         + f"<td>{ratio:.2f}</td></tr>")
    table.append("</table>")

    html = ["<html><body style='font-family:sans-serif;max-width:1100px;margin:auto'>",
            "<h1>Pixel bias comparison: 10 uA vs 20 uA</h1>",
            f"<p>Saturated spot: centre row {spot['cy']}, x {spot['x0']}-{spot['x1']} (radius ~{spot['r']} px). "
            f"Profiles average rows {spot['cy'] - args.half_band} to {spot['cy'] + args.half_band}, "
            f"{args.span} px out from each edge. Excess = mean over {NEAR[0]}-{NEAR[1]} px from the edge minus the "
            f"mean over the outermost {FAR_WIDTH} px. Top plots are log-scale; the difference plots start "
            f"{NEAR[0]} px from the edge.</p>",
            f"<img src='data:image/png;base64,{overview}'>", "<h2>Summary</h2>",
            f"<img src='data:image/png;base64,{summary}'>", "".join(table)]
    for e, img in figs:
        html.append(f"<h2>{e} ms</h2><img src='data:image/png;base64,{img}'>")
    (args.corrected / "report.html").write_text("".join(html) + "</body></html>")
    print("report:", args.corrected / "report.html")


if __name__ == "__main__":
    main()
