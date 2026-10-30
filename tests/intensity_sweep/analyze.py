"""How the dark box around the spot scales with source intensity, relative to spot saturation.

  python -m tests.intensity_sweep.analyze 500ms
  python -m tests.intensity_sweep.analyze 1000ms_20uA

Reads output/intensity_sweep/<sweep>/<P>pc_dc.tif from tests.intensity_sweep.correct, plus the
raw stacks in data/intensity_sweep/<sweep>/ (for saturated-pixel counts). Writes report.html
and sweep.csv next to the corrected images.

Two contrasts (negative = darker than surroundings), each a 10% trimmed mean difference:
- box contrast: inner box interior (spot excluded) minus its surround inside the light-box footprint.
- footprint contrast: a band just inside the footprint's right edge minus a band just outside it.

Overdrive (spot peak / saturation) does not assume a linear source. The brightest unsaturated
image gives the spot's radial profile P(r). A saturated image whose saturated disc has radius R
has a flux f with f*P(R) = saturation, so its overdrive is P(0)/P(R). This assumes the spot
shape does not change with intensity, and becomes noisy once R reaches the profile's faint tail.

Box/surround/quadrant positions are per sweep (REGIONS); check them on the overview image and
override with --box/--surround-exclude/--quadrant.
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
from scipy.stats import trim_mean

SAT = 2**14 - 1
SPOT_EXCLUDE_PX = 150  # radius around the spot left out of the box statistic
SAT_SEARCH_PX = 150  # saturated pixels are counted within this radius of the spot

# x0, y0, x1, y1 in pixels, fitted by eye to each sweep's images
REGIONS = {
    "500ms": {"box": [260, 470, 840, 1090], "surround_exclude": [200, 410, 900, 1150], "quadrant": [20, 20, 1010, 1480],
              "footprint_in": [850, 300, 950, 1300], "footprint_out": [1060, 300, 1160, 1300]},
    "1000ms_20uA": {"box": [170, 510, 740, 1040], "surround_exclude": [90, 420, 830, 1130], "quadrant": [100, 100, 980, 1450],
                    "footprint_in": [850, 300, 950, 1300], "footprint_out": [1060, 300, 1160, 1300]},
}
REGION_ARGS = ("box", "surround_exclude", "quadrant", "footprint_in", "footprint_out")
CONTRASTS = {"box_contrast_adu": ("inner box − surround", "k"), "footprint_contrast_adu": ("footprint − outside", "tab:purple")}


def tmean(a):
    return float(trim_mean(a, 0.1))


def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def spot_centre(img):
    """Centroid of the largest bright blob; median filter removes hot pixels."""
    smooth = ndimage.median_filter(img, size=5)
    labels, n = ndimage.label(smooth > 0.5 * smooth.max())
    biggest = 1 + int(np.argmax(ndimage.sum(np.ones_like(labels), labels, range(1, n + 1))))
    return ndimage.center_of_mass(labels == biggest)


def radial_profile(img, cy, cx, yy, xx, rmax=100):
    """Median over 1-px rings around the spot, minus the background just beyond rmax."""
    r = np.hypot(yy - cy, xx - cx)
    ri = r.astype(int)
    m = ri < rmax
    bg = np.median(img[(r >= rmax) & (r < rmax + 40)])
    return ndimage.median(img[m], labels=ri[m], index=np.arange(rmax)) - bg


def in_rect(yy, xx, r):
    x0, y0, x1, y1 = r
    return (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)


def outline(ax, r, color, label):
    x0, y0, x1, y1 = r
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=1.2, label=label))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("sweep", nargs="?", default="500ms", help="sweep folder name, e.g. 500ms or 1000ms_20uA")
    p.add_argument("--corrected", type=Path, default=Path("output/intensity_sweep"))
    p.add_argument("--data", type=Path, default=Path("data/intensity_sweep"))
    rect = {"type": int, "nargs": 4, "metavar": ("X0", "Y0", "X1", "Y1")}
    p.add_argument("--box", **rect, help="box interior used for the statistic")
    p.add_argument("--surround-exclude", **rect, help="box plus its edges, left out of the surround")
    p.add_argument("--quadrant", **rect, help="readout quadrant holding the box; the surround is taken inside it")
    p.add_argument("--footprint-in", **rect, help="band just inside the light-box footprint edge")
    p.add_argument("--footprint-out", **rect, help="band just outside the footprint edge")
    args = p.parse_args()
    for key, val in REGIONS.get(args.sweep, {}).items():
        if getattr(args, key) is None:
            setattr(args, key, val)
    if any(getattr(args, k) is None for k in REGION_ARGS):
        p.error(f"no default regions for '{args.sweep}'; pass all of " + ", ".join("--" + k.replace("_", "-") for k in REGION_ARGS))
    corrected, data = args.corrected / args.sweep, args.data / args.sweep

    files = {int(m.group(1)): f for f in corrected.glob("*pc_dc.tif") if (m := re.match(r"(\d+)pc_", f.name))}
    pcs = sorted(files)
    imgs = {pc: tifffile.imread(files[pc]).astype(float) for pc in pcs}
    h, w = imgs[pcs[0]].shape
    yy, xx = np.mgrid[0:h, 0:w]
    box_m = in_rect(yy, xx, args.box)
    surround_m = in_rect(yy, xx, args.quadrant) & ~in_rect(yy, xx, args.surround_exclude)
    fp_in_m, fp_out_m = in_rect(yy, xx, args.footprint_in), in_rect(yy, xx, args.footprint_out)

    rows = []
    for pc in pcs:
        img = imgs[pc]
        raw = tifffile.imread(next(data.glob(f"{pc}_pc_*.tif"))).astype(float)
        raw = raw.reshape(-1, *raw.shape[-2:]).mean(0)
        cy, cx = spot_centre(img)
        r = np.hypot(yy - cy, xx - cx)
        box_lvl = tmean(img[box_m & (r > SPOT_EXCLUDE_PX)])
        sur_lvl = tmean(img[surround_m])
        rows.append({"pc": pc, "spot_y": cy, "spot_x": cx, "core_adu": img[r < 5].mean() - 100,
                     "spot_sat_px": int((raw[r < SAT_SEARCH_PX] >= SAT).sum()), "box_adu": box_lvl - 100,
                     "surround_adu": sur_lvl - 100, "box_contrast_adu": box_lvl - sur_lvl,
                     "outside_adu": tmean(img[fp_out_m]) - 100,
                     "footprint_contrast_adu": tmean(img[fp_in_m]) - tmean(img[fp_out_m])})

    unsat = [x for x in rows if x["spot_sat_px"] == 0]
    sat = [x for x in rows if x["spot_sat_px"] > 0]
    first_sat = sat[0]["pc"] if sat else None
    sat_core = np.median([x["core_adu"] for x in sat]) if sat else np.nan
    ref = max(unsat, key=lambda x: x["core_adu"]) if unsat else None
    prof = radial_profile(imgs[ref["pc"]], ref["spot_y"], ref["spot_x"], yy, xx) if ref else None
    for x in rows:
        if x["spot_sat_px"] == 0:
            x["sat_radius_px"] = 0.0
            x["overdrive"] = x["core_adu"] / sat_core if sat else np.nan
        else:
            x["sat_radius_px"] = np.sqrt(x["spot_sat_px"] / np.pi)
            p_r = np.interp(x["sat_radius_px"], np.arange(len(prof)), prof) if ref else np.nan
            x["overdrive"] = prof[:2].mean() / p_r if ref and p_r > 0 else np.nan
    floor = {k: 3 * np.std([x[k] for x in unsat]) if len(unsat) > 1 else np.nan for k in CONTRASTS}
    base = {k: np.mean([x[k] for x in unsat]) if unsat else 0.0 for k in CONTRASTS}

    with open(corrected / "sweep.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    # overview: brightest image with regions
    top = pcs[-1]
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ndimage.median_filter(imgs[top], 5)
    ax.imshow(im, cmap="gray", vmin=np.percentile(im, 2), vmax=np.percentile(im, 98))
    outline(ax, args.box, "red", "box (statistic)")
    outline(ax, args.surround_exclude, "orange", "excluded from surround")
    outline(ax, args.quadrant, "lime", "quadrant (surround inside)")
    outline(ax, args.footprint_in, "violet", "footprint edge: inside")
    outline(ax, args.footprint_out, "cyan", "footprint edge: outside")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"{args.sweep}: {top}%")
    overview = png(fig)

    # scaling plots
    fig, ax = plt.subplots(1, 4, figsize=(22, 4.5))
    sc = ax[3].scatter([x["outside_adu"] for x in rows], [x["footprint_contrast_adu"] for x in rows],
                       c=np.log10([max(x["overdrive"], 1e-3) for x in rows]), cmap="viridis")
    for x in rows:
        ax[3].annotate(f"{x['pc']}%", (x["outside_adu"], x["footprint_contrast_adu"]), fontsize=7, xytext=(3, 3),
                       textcoords="offset points")
    fig.colorbar(sc, ax=ax[3], label="log10 overdrive")
    ax[3].set_xlabel("level outside footprint (ADU)"); ax[3].set_ylabel("footprint − outside (ADU)")
    ax[3].set_title("footprint contrast vs outside level\n(slope ≈ −1 means only the outside is changing)")
    ax[0].plot([x["pc"] for x in rows], [x["overdrive"] for x in rows], "o-")
    ax[0].axhline(1, c="r", lw=0.7, label="saturation")
    ax[0].set_yscale("log"); ax[0].set_xlabel("source intensity (%)"); ax[0].set_ylabel("spot peak / saturation")
    ax[0].set_title(f"estimated overdrive (profile from {ref['pc']}%)" if ref else "overdrive (no unsaturated image)")
    ax[0].legend()
    for a, key, xl in ((ax[1], "pc", "source intensity (%)"), (ax[2], "overdrive", "estimated overdrive (log)")):
        xs = [x[key] for x in rows]
        for k, (label, color) in CONTRASTS.items():
            a.plot(xs, [x[k] for x in rows], "o-", c=color, label=label)
            if np.isfinite(floor[k]):
                a.axhspan(base[k] - floor[k], base[k] + floor[k], color=color, alpha=0.12, label="±3σ unsaturated")
        a.axvline(first_sat if key == "pc" else 1.0, c="r", lw=0.7, label="spot saturates")
        if key == "overdrive":
            a.set_xscale("log")
        a.axhline(0, c="gray", lw=0.5)
        a.set_xlabel(xl); a.set_ylabel("contrast (ADU)"); a.set_title("dark box contrast"); a.legend(fontsize=8)
    scaling = png(fig)

    # profiles through the spot, surround-subtracted, for a spread of intensities
    pick = sorted(set(pcs[::max(1, len(pcs) // 6)] + [pcs[-1]] + ([first_sat] if first_sat else [])))
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
    cmap = plt.get_cmap("viridis")
    for i, pc in enumerate(pick):
        x = next(r for r in rows if r["pc"] == pc)
        cy, cx = int(round(x["spot_y"])), int(round(x["spot_x"]))
        sm = ndimage.median_filter(imgs[pc], 5) - (x["surround_adu"] + 100)
        c = cmap(i / max(1, len(pick) - 1))
        ax[0].plot(sm[cy - 50:cy + 51, args.quadrant[0]:args.quadrant[2]].mean(0), c=c, lw=0.8, label=f"{pc}%")
        ax[1].plot(sm[args.quadrant[1]:args.quadrant[3], cx - 50:cx + 51].mean(1), c=c, lw=0.8, label=f"{pc}%")
    for a, (lo, hi), t in ((ax[0], (args.box[0], args.box[2]), "horizontal (rows ±50 around spot)"),
                           (ax[1], (args.box[1], args.box[3]), "vertical (cols ±50 around spot)")):
        off = args.quadrant[0] if a is ax[0] else args.quadrant[1]
        a.axvspan(lo - off, hi - off, color="red", alpha=0.07, label="box")
        a.set_ylim(-30, 40); a.axhline(0, c="gray", lw=0.5)
        a.set_xlabel("px from quadrant edge"); a.set_ylabel("ADU − surround"); a.set_title(t); a.legend(fontsize=7, ncol=2)
    profiles = png(fig)

    # image grid at a common, surround-referenced contrast
    n = len(pcs)
    cols = 4 if n <= 8 else 5
    fig, ax = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(3.2 * cols, 3.4 * np.ceil(n / cols)))
    qx0, qy0, qx1, qy1 = args.quadrant
    for a, x in zip(ax.flat, rows):
        q = ndimage.median_filter(imgs[x["pc"]][qy0:qy1, qx0:qx1], 5) - (x["surround_adu"] + 100)
        a.imshow(q, cmap="gray", vmin=-20, vmax=20)
        a.set_title(f"{x['pc']}%  contrast {x['box_contrast_adu']:+.1f}", fontsize=9)
        a.axis("off")
    for a in list(ax.flat)[n:]:
        a.axis("off")
    grid_png = png(fig)

    lo, hi = rows[0], max(rows, key=lambda x: x["overdrive"] if np.isfinite(x["overdrive"]) else -1)
    verdicts = []
    for k, (label, _) in CONTRASTS.items():
        vals = [x[k] for x in rows]
        v = (f"{label.capitalize()}: {lo[k]:+.1f} ADU at {lo['pc']}% (overdrive ≈ {lo['overdrive']:.2g}), "
             f"{hi[k]:+.1f} ADU at {hi['pc']}% (overdrive ≈ {hi['overdrive']:.2g}); range {min(vals):+.1f} to {max(vals):+.1f}.")
        if not np.isfinite(floor[k]):
            v += " Fewer than 2 unsaturated images, so no noise band to test against."
        else:
            beyond = [x for x in sat if abs(x[k] - base[k]) > floor[k]]
            v += (f" First outside the ±3σ band of the unsaturated images (±{floor[k]:.1f} ADU) at {beyond[0]['pc']}% "
                  f"(overdrive ≈ {beyond[0]['overdrive']:.2g})." if beyond else
                  f" No saturated image outside the ±3σ band of the unsaturated images (±{floor[k]:.1f} ADU).")
        verdicts.append(v)
    verdict = "<br>".join(verdicts)
    table = ["<table border=1 cellpadding=3 style='border-collapse:collapse;font-size:13px'><tr>"
             "<th>%</th><th>core (ADU)</th><th>sat px</th><th>sat radius</th><th>overdrive</th><th>box</th><th>surround</th><th>box contrast</th><th>outside footprint</th><th>footprint contrast</th></tr>"]
    table += [f"<tr><td>{x['pc']}</td><td>{x['core_adu']:.0f}</td><td>{x['spot_sat_px']}</td><td>{x['sat_radius_px']:.1f}</td><td>{x['overdrive']:.3g}</td>"
              f"<td>{x['box_adu']:.1f}</td><td>{x['surround_adu']:.1f}</td><td>{x['box_contrast_adu']:+.1f}</td>"
              f"<td>{x['outside_adu']:.1f}</td><td>{x['footprint_contrast_adu']:+.1f}</td></tr>" for x in rows]
    table.append("</table>")
    html = ["<html><body style='font-family:sans-serif;max-width:1250px;margin:auto'>",
            f"<h1>Intensity sweep {args.sweep}: dark box vs source intensity</h1>",
            f"<p><b>{verdict}</b> Spot first saturates at {first_sat}%.</p>",
            "<p>Box, surround and core are dark-corrected ADU with the 100 offset removed. Overdrive (spot peak / "
            "saturation) is the core level for unsaturated images. For saturated images it is P(0)/P(R), where P is "
            "the radial profile of the brightest unsaturated image and R the radius of the saturated disc. This does "
            "not assume a linear source, but it does assume a fixed spot shape, and it is noisy once R reaches the "
            "profile's faint tail. All images share one dark frame, so offset drift between images shows up in both "
            "regions of each contrast and cancels. Stray ambient light does not cancel: it reaches the sensor outside "
            "the light-box footprint but not under it.</p>",
            f"<img src='data:image/png;base64,{overview}'>",
            "<h2>Scaling</h2>", f"<img src='data:image/png;base64,{scaling}'>",
            "<h2>Profiles through the spot</h2>", f"<img src='data:image/png;base64,{profiles}'>",
            "<h2>All intensities (surround-subtracted, ±20 ADU)</h2>", f"<img src='data:image/png;base64,{grid_png}'>",
            "<h2>Table</h2>", "".join(table), "</body></html>"]
    (corrected / "report.html").write_text("".join(html))
    print(verdict.replace("<br>", "\n"))
    print("report:", corrected / "report.html")


if __name__ == "__main__":
    main()
