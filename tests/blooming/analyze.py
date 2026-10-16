"""Analyse a blooming capture session and write an HTML report.

  python -m tests.blooming.analyze output/blooming/M4/<timestamp> --pitch-um 74.8

Writes into the session folder: rois.json (editable; reused on later runs),
metrics.csv and report.html (per-exposure image + surface plot of every ROI).
"""
import argparse
import base64
import csv
import io
import json
from html import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

OFFSET = 100
SAT = 2**14 - 1
ROI_NAMES = ["left", "right", "top", "bottom"]


def load(session):
    meta = json.loads((session / "meta.json").read_text())
    caps = {(c["kind"], c["exposure_ms"]): c for c in meta["captures"]}
    exps = sorted({e for k, e in caps if k == "bright"} & {e for k, e in caps if k == "dark"})
    data = {}
    for e in exps:
        bright = np.load(session / caps[("bright", e)]["file"]).astype(float)
        dark = np.load(session / caps[("dark", e)]["file"]).astype(float)
        corrected = np.clip(bright.mean(0) - dark.mean(0) + OFFSET, 0, SAT)
        data[e] = {"img": corrected, "frames": bright - dark.mean(0) + OFFSET,
                   "temp": caps[("bright", e)]["temp_after"],
                   "sat": caps[("bright", e)]["saturated_pixels"]}
    return meta, exps, data


def make_rois(img, args):
    """Place left/right/top/bottom ROIs around the core, plus a control in the farthest corner."""
    h, w = img.shape
    ys, xs = np.nonzero(img > 0.5 * img.max())
    cx, cy = xs.mean(), ys.mean()
    d = round(args.roi_dist_mm * 1000 / args.pitch_um)
    s = round(args.roi_size_mm * 1000 / args.pitch_um)
    centers = {"left": (cx - d, cy), "right": (cx + d, cy), "top": (cx, cy - d), "bottom": (cx, cy + d)}
    rois = {}
    for name, (x, y) in centers.items():
        x0, y0 = round(x - s / 2), round(y - s / 2)
        if x0 < 0 or y0 < 0 or x0 + s > w or y0 + s > h:
            raise SystemExit(f"{name} ROI falls outside the image; reduce --roi-dist-mm/--roi-size-mm")
        rois[name] = [x0, y0, x0 + s, y0 + s]
    corners = [(0, 0), (w - s, 0), (0, h - s), (w - s, h - s)]
    x0, y0 = max(corners, key=lambda c: (c[0] + s / 2 - cx) ** 2 + (c[1] + s / 2 - cy) ** 2)
    rois["control"] = [x0, y0, x0 + s, y0 + s]
    return {"core_xy": [cx, cy], "rois": rois}


def crop(img, box):
    x0, y0, x1, y1 = box
    return img[y0:y1, x0:x1]


def block_mean(a, max_side=60):
    f = max(1, int(np.ceil(max(a.shape) / max_side)))
    h, w = a.shape[0] // f * f, a.shape[1] // f * f
    return a[:h, :w].reshape(h // f, f, w // f, f).mean((1, 3)), f


def fit_through_origin(t, y):
    t, y = np.asarray(t, float), np.asarray(y, float)
    return float((t * y).sum() / (t * t).sum()) if len(t) else np.nan


def metrics(exps, data, geo):
    cx, cy = (int(round(v)) for v in geo["core_xy"])
    rois = geo["rois"]
    # unsaturated exposures: raw core patch well below full scale
    core = {e: data[e]["img"][cy - 3:cy + 4, cx - 3:cx + 4].mean() for e in exps}
    unsat = [e for e in exps if data[e]["img"][cy - 3:cy + 4, cx - 3:cx + 4].max() < 0.95 * SAT]
    core_slope = fit_through_origin(unsat, [core[e] for e in unsat])
    diffs = {n: {e: crop(data[e]["img"], rois[n]).mean() - crop(data[e]["img"], rois["control"]).mean()
                 for e in exps} for n in ROI_NAMES}
    glare = {n: fit_through_origin(unsat, [diffs[n][e] for e in unsat]) for n in ROI_NAMES}
    rows = []
    for e in exps:
        ctrl = crop(data[e]["img"], rois["control"])
        noise = ctrl.std()
        for n in ROI_NAMES:
            r = crop(data[e]["img"], rois[n])
            diff = diffs[n][e]
            excess = diff - glare[n] * e if np.isfinite(glare[n]) else np.nan
            bloom = excess if np.isfinite(excess) else diff
            se = np.sqrt(r.var() / r.size + ctrl.var() / ctrl.size)
            rows.append({
                "exposure_ms": e, "roi": n,
                "overdrive": core_slope * e / SAT if np.isfinite(core_slope) else np.nan,
                "diff_adu": diff, "glare_adu": glare[n] * e, "excess_adu": excess,
                "noise_adu": noise, "cnr": bloom / noise, "t_stat": bloom / se,
                "temp": data[e]["temp"], "saturated_px": data[e]["sat"]})
    return rows, unsat


def fig_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def exposure_figure(e, d, geo, rows, zlim):
    rois = geo["rois"]
    img = d["img"]
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img, cmap="gray", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
    for n, (x0, y0, x1, y1) in rois.items():
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               ec="lime" if n == "control" else "red", lw=1.5))
        ax.text(x0, y0 - 3, n, color="lime" if n == "control" else "red", fontsize=8)
    ax.set_title(f"{e:g} ms (mean of {len(d['frames'])} frames, dark-corrected)")
    ax.axis("off")
    ctrl_mean = crop(img, rois["control"]).mean()
    blocks = {n: block_mean(crop(img, rois[n]))[0] for n in ROI_NAMES}
    lo, hi = zlim or (min(min(b.min() for b in blocks.values()), ctrl_mean),
                      max(max(b.max() for b in blocks.values()), ctrl_mean))
    for n, cell in zip(ROI_NAMES, [gs[0, 1], gs[0, 2], gs[1, 1], gs[1, 2]]):
        z = blocks[n]
        a3 = fig.add_subplot(cell, projection="3d")
        X, Y = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))
        a3.plot_surface(X, Y, z, cmap="viridis", linewidth=0, antialiased=False)
        a3.plot_surface(X, Y, np.full_like(z, ctrl_mean), color="red", alpha=0.25, linewidth=0)
        a3.set_zlim(lo, hi)
        a3.set_title(f"{n} (red plane = control mean)", fontsize=9)
        a3.set_zlabel("ADU")
        a3.set_xticks([]); a3.set_yticks([])
    at = fig.add_subplot(gs[1, 0]); at.axis("off")
    lines = [f"temp {d['temp']}   saturated px {d['sat']}", f"overdrive {rows[0]['overdrive']:.2f}", "",
             f"{'roi':7s}{'diff':>8s}{'excess':>8s}{'CNR':>7s}{'t':>7s}"]
    lines += [f"{r['roi']:7s}{r['diff_adu']:8.2f}{r['excess_adu']:8.2f}{r['cnr']:7.2f}{r['t_stat']:7.1f}" for r in rows]
    at.text(0, 1, "\n".join(lines), family="monospace", va="top", fontsize=9)
    return fig_png(fig)


def summary_figure(rows):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for n in ROI_NAMES:
        rs = [r for r in rows if r["roi"] == n]
        e = [r["exposure_ms"] for r in rs]
        axes[0].plot(e, [r["diff_adu"] for r in rs], "o-", label=n)
        axes[1].plot(e, [r["excess_adu"] for r in rs], "o-", label=n)
    for a, t in zip(axes, ["ROI - control (ADU)", "excess over glare fit (ADU)"]):
        a.set_xscale("log"); a.set_xlabel("exposure (ms)"); a.set_title(t); a.axhline(0, c="k", lw=.5); a.legend()
    return fig_png(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("session", type=lambda s: __import__("pathlib").Path(s))
    p.add_argument("--pitch-um", type=float, help="pixel pitch; needed when creating rois.json")
    p.add_argument("--roi-dist-mm", type=float, default=3.0, help="core-to-ROI-centre distance")
    p.add_argument("--roi-size-mm", type=float, default=2.0)
    p.add_argument("--regen-rois", action="store_true")
    p.add_argument("--zlim", type=float, nargs=2, help="fixed z-axis limits (ADU) for all surface plots")
    args = p.parse_args()

    meta, exps, data = load(args.session)
    roi_file = args.session / "rois.json"
    if roi_file.exists() and not args.regen_rois:
        geo = json.loads(roi_file.read_text())
    else:
        if not args.pitch_um:
            p.error("--pitch-um is required to create rois.json")
        geo = make_rois(data[exps[-1]]["img"], args)
        roi_file.write_text(json.dumps(geo, indent=2))
    rows, unsat = metrics(exps, data, geo)
    if len(unsat) < 2:
        print(f"warning: only {len(unsat)} unsaturated exposure(s); glare fit and overdrive are unreliable")

    with open(args.session / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    parts = [f"<h1>Blooming report: {escape(meta['sensor'])}</h1>",
             f"<p>interface {meta['iface']}, full well {meta['full_well']}, DDS {meta['dds']}, "
             f"{meta['frames_per_trigger']} frames/trigger averaged. Unsaturated exposures used for glare fit: "
             f"{unsat}. Blooming = excess over glare fit; CNR = excess / control-ROI noise; "
             f"t is approximate (pixels are correlated).</p>",
             f"<img src='data:image/png;base64,{summary_figure(rows)}'>"]
    for e in exps:
        parts.append(f"<h2>{e:g} ms</h2><img src='data:image/png;base64,"
                     f"{exposure_figure(e, data[e], geo, [r for r in rows if r['exposure_ms'] == e], args.zlim)}'>")
    (args.session / "report.html").write_text(
        "<html><body style='font-family:sans-serif;max-width:1200px;margin:auto'>" + "".join(parts) + "</body></html>")
    print("report:", args.session / "report.html")


if __name__ == "__main__":
    main()
