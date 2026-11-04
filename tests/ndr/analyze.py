"""Analyse an NDR capture session and write report.html, summary.csv and slope maps.

  python -m tests.ndr.analyze output/ndr/<sensor>/<timestamp> [--roi X0 Y0 X1 Y1]

Checks, per light state:
- Non-destructive: the ROI mean should rise read after read (a destructive read would reset it).
  The minimum read-to-read increment is reported as a z-score against its own noise.
- Linearity: straight-line fit of ROI mean vs read time; R² and the worst residual.
- Agreement with destructive readout: NDR slope vs the slope through the reference captures,
  and each read vs the reference at the same exposure.
- Noise: std of second differences (r[k+1] - 2 r[k] + r[k-1]) / sqrt(6). A linear ramp cancels, so
  spatial variation in signal rate doesn't leak in; for a dark ramp this is the read noise.
- Per-pixel up-the-ramp slope map (ADU/ms), using only the reads before a pixel saturates.
- SNR table (needs >= 2 repeats): signal and temporal noise of three estimators of the same
  exposure, each pooled over ROI pixels as mean(signal) / sqrt(mean(per-pixel variance across repeats)):
    ramp   slope fit through all reads x integration time T (last read time)
    CDS    last read - first read
    single one destructive reference frame at T minus the dark reference (needs dark refs and >= 2 ref frames)
  Pooling the per-pixel variance across repeats keeps fixed-pattern noise out of the noise term.
- Sensor metrics: read noise (dark 2nd-difference noise), conversion gain from the CDS photon
  transfer var = gain * mean + 2 * read_noise^2 (bright), dark current, bias and dynamic range.
ROI statistics use only pixels that stay below saturation in every NDR read of that light state.
"""
import argparse
import base64
import csv
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

SAT = 2**14 - 1
SAT_FIT_LIMIT = 0.95 * SAT  # reads at or above this are left out of per-pixel fits


def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=85, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def stack(path):
    a = tifffile.imread(path)
    return a.reshape(-1, *a.shape[-2:])


def slope_map(reads, t, chunk=256):
    """Per-pixel least-squares slope (ADU/ms) and number of reads used, stopping at the first saturated read."""
    n, h, w = reads.shape
    slope = np.full((h, w), np.nan, np.float32)
    used = np.zeros((h, w), np.int16)
    tt = np.asarray(t, np.float64)[:, None, None]
    for r0 in range(0, h, chunk):
        y = reads[:, r0:r0 + chunk].astype(np.float64)
        wt = np.cumprod(y < SAT_FIT_LIMIT, axis=0)
        k = wt.sum(0)
        st, sy = (wt * tt).sum(0), (wt * y).sum(0)
        stt, sty = (wt * tt * tt).sum(0), (wt * tt * y).sum(0)
        den = k * stt - st ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            slope[r0:r0 + chunk] = np.where((k >= 2) & (den > 0), (k * sty - st * sy) / den, np.nan)
        used[r0:r0 + chunk] = k
    return slope, used


def pooled_snr(samples):
    """samples (repeats, pixels): mean signal, pooled temporal noise and their ratio."""
    if len(samples) < 2:
        return np.nan, np.nan, np.nan
    signal = float(samples.mean())
    noise = float(np.sqrt(samples.var(axis=0, ddof=1).mean()))
    return signal, noise, signal / noise if noise > 0 else np.nan


def pixel_slopes(px, t):
    """Least-squares slope per pixel; px is (reads, pixels)."""
    tc = t - t.mean()
    return (tc[:, None] * px).sum(0) / (tc ** 2).sum()


def line_fit(t, y):
    a, b = np.polyfit(t, y, 1)
    resid = np.asarray(y) - (a * np.asarray(t) + b)
    r2 = 1 - (resid ** 2).sum() / ((np.asarray(y) - np.mean(y)) ** 2).sum() if np.ptp(y) else np.nan
    return a, b, resid, r2


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("session", type=Path)
    p.add_argument("--roi", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                   help="analysis region; default is the central quarter of the frame")
    args = p.parse_args()

    meta = json.loads((args.session / "meta.json").read_text())
    caps = meta["captures"]
    states = [s for s in ("dark", "bright") if any(c["light"] == s for c in caps)]
    first = stack(args.session / caps[0]["file"])
    h, w = first.shape[-2:]
    x0, y0, x1, y1 = args.roi or (w // 4, h // 4, 3 * w // 4, 3 * h // 4)
    roi = (slice(y0, y1), slice(x0, x1))

    rows, sections, snr_rows, sensor = [], [], [], {}
    for state in states:
        ndr_caps = [c for c in caps if c["kind"] == "ndr" and c["light"] == state]
        ref_caps = sorted((c for c in caps if c["kind"] == "reference" and c["light"] == state),
                          key=lambda c: c["exposure_ms"])
        ndr_reads = [stack(args.session / c["file"]) for c in ndr_caps]
        ok = np.all([np.all(r[(slice(None), *roi)] < SAT_FIT_LIMIT, axis=0) for r in ndr_reads], axis=0)
        ref_t = [c["exposure_ms"] for c in ref_caps]
        ref_mean = [stack(args.session / c["file"])[(slice(None), *roi)][:, ok].mean() for c in ref_caps]
        ref_slope = line_fit(ref_t, ref_mean)[0] if len(ref_t) >= 2 else np.nan

        # SNR of ramp fit, CDS and a single destructive frame, all at the same integration time T
        t_all = np.asarray(ndr_caps[0]["read_times_ms"], float)
        T = t_all[-1]
        px = [r[(slice(None), *roi)][:, ok].astype(np.float64) for r in ndr_reads]
        slopes_px = np.stack([pixel_slopes(q, t_all) for q in px])
        ramp = pooled_snr(slopes_px * T)
        cds_samples = np.stack([q[-1] - q[0] for q in px])
        cds = pooled_snr(cds_samples)
        single = (np.nan, np.nan, np.nan)
        ref_T = next((c for c in ref_caps if abs(c["exposure_ms"] - T) < 1e-6), None)
        dark_T = next((c for c in caps if c["kind"] == "reference" and c["light"] == "dark"
                       and abs(c["exposure_ms"] - T) < 1e-6), None)
        if state == "bright" and ref_T and dark_T:
            b = stack(args.session / ref_T["file"])[(slice(None), *roi)][:, ok].astype(np.float64)
            d = stack(args.session / dark_T["file"])[(slice(None), *roi)][:, ok].astype(np.float64)
            if len(b) >= 2:
                sig = (b.mean(0) - d.mean(0)).mean()
                noise = np.sqrt(b.var(axis=0, ddof=1).mean())
                single = (sig, noise, sig / noise)
        mean_slope = slopes_px.mean(0)
        snr_rows.append({"light": state, "repeats": len(px), "reads": len(t_all), "integration_ms": T,
                         "ramp_signal_adu": ramp[0], "ramp_noise_adu": ramp[1], "snr_ramp": ramp[2],
                         "cds_signal_adu": cds[0], "cds_noise_adu": cds[1], "snr_cds": cds[2],
                         "single_signal_adu": single[0], "single_noise_adu": single[1], "snr_single": single[2],
                         "ramp_vs_cds": ramp[2] / cds[2], "ramp_vs_single": ramp[2] / single[2],
                         "spatial_spread_pct": 100 * mean_slope.std() / abs(mean_slope.mean()) if mean_slope.mean() else np.nan})
        if state == "dark":
            sensor["dark_current_adu_per_s"] = float(mean_slope.mean() * 1000)
            sensor["bias_adu"] = float(np.mean([q[0].mean() for q in px]) - mean_slope.mean() * t_all[0])
            if len(t_all) >= 3:
                sensor["read_noise_adu"] = float(np.median(np.concatenate(
                    [np.diff(q, n=2, axis=0).std(1) / np.sqrt(6) for q in px])))
        else:
            sensor["_cds"] = (cds[0], cds[1])

        fig, ax = plt.subplots(2, 2, figsize=(13, 8))
        slopes = []
        for c, reads in zip(ndr_caps, ndr_reads):
            t = np.asarray(c["read_times_ms"], float)
            m = reads[(slice(None), *roi)][:, ok].astype(float)  # (reads, unsaturated ROI pixels)
            mean = m.mean(1)
            diffs = np.diff(m, axis=0)
            inc = diffs.mean(1)
            inc_z = inc / (diffs.std(1) / np.sqrt(m.shape[1]))
            noise = np.diff(m, n=2, axis=0).std(1) / np.sqrt(6) if len(m) >= 3 else np.full(1, np.nan)
            a, b, resid, r2 = line_fit(t, mean)
            ref_at = np.interp(t, ref_t, ref_mean) if ref_t else np.full_like(t, np.nan)
            smap, used = slope_map(reads, t)
            slopes.append(smap)
            rows.append({"light": state, "repeat": c["repeat"], "reads": len(reads),
                         "ndr_slope_adu_per_ms": a, "ref_slope_adu_per_ms": ref_slope,
                         "slope_ratio": a / ref_slope if ref_slope else np.nan, "intercept_adu": b, "r2": r2,
                         "max_abs_resid_adu": np.abs(resid).max(), "min_increment_adu": inc.min(),
                         "min_increment_z": inc_z.min(), "noise_adu": float(np.median(noise)),
                         "max_read_vs_ref_adu": float(np.nanmax(np.abs(mean - ref_at))) if ref_t else np.nan,
                         "pixels_saturating": int((used < len(reads)).sum()), "roi_pixels_used": int(ok.sum())})
            lbl = f"rep {c['repeat']}"
            ax[0, 0].plot(t, mean, "o-", ms=3, label=lbl)
            ax[0, 1].plot(t, resid, "o-", ms=3, label=lbl)
            ax[1, 0].plot(t[1:], inc, "o-", ms=3, label=lbl)
            ax[1, 1].plot(t[1:-1], noise, "o-", ms=3, label=lbl)
        if ref_t:
            ax[0, 0].plot(ref_t, ref_mean, "ks", mfc="none", label="destructive reference")
        for a_, title, yl in ((ax[0, 0], "ROI mean vs read time", "ADU"),
                              (ax[0, 1], "residual from straight-line fit", "ADU"),
                              (ax[1, 0], "increment per read (should stay > 0)", "ADU"),
                              (ax[1, 1], "noise (std of 2nd difference / √6)", "ADU")):
            a_.set_title(f"{state}: {title}"); a_.set_xlabel("time since reset (ms)"); a_.set_ylabel(yl)
            a_.axhline(0, c="gray", lw=0.5); a_.legend(fontsize=7)
        ramp_png = png(fig)

        smed = np.nanmedian(np.stack(slopes), axis=0)
        tifffile.imwrite(args.session / f"slope_{state}.tif", smed.astype(np.float32))
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        lo, hi = np.nanpercentile(smed, [1, 99])
        im = ax[0].imshow(smed, cmap="magma", vmin=lo, vmax=hi)
        ax[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="cyan", lw=1))
        fig.colorbar(im, ax=ax[0], label="ADU/ms")
        ax[0].set_title(f"{state}: per-pixel slope (median of repeats)")
        ax[1].hist(smed[np.isfinite(smed)].ravel(), bins=200, range=(lo, hi), color="gray")
        ax[1].set_yscale("log")
        ax[1].set_xlabel("ADU/ms"); ax[1].set_title("slope histogram (1-99 percentile range)")
        map_png = png(fig)
        sections.append((state, ramp_png, map_png))

    with open(args.session / "summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    # sensor metrics: photon transfer on the bright CDS needs the dark read noise
    rn = sensor.get("read_noise_adu", np.nan)
    cds_mean, cds_noise = sensor.pop("_cds", (np.nan, np.nan))
    gain = (cds_noise ** 2 - 2 * rn ** 2) / cds_mean if cds_mean and cds_mean > 0 else np.nan
    gain = gain if gain > 0 else np.nan
    sensor["conversion_gain_adu_per_e"] = gain
    sensor["read_noise_e"] = rn / gain
    sensor["dark_current_e_per_s"] = sensor.get("dark_current_adu_per_s", np.nan) / gain
    sensor["dynamic_range_db"] = 20 * np.log10((SAT - sensor.get("bias_adu", np.nan)) / rn)
    with open(args.session / "snr.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(snr_rows[0]))
        wr.writeheader()
        wr.writerows(snr_rows)
    with open(args.session / "sensor_metrics.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["metric", "value"])
        wr.writerows(sensor.items())

    cols = ["light", "repeat", "ndr_slope_adu_per_ms", "ref_slope_adu_per_ms", "slope_ratio", "r2",
            "max_abs_resid_adu", "min_increment_z", "noise_adu", "max_read_vs_ref_adu", "pixels_saturating",
            "roi_pixels_used"]
    fmt = lambda v: ("n/a" if not np.isfinite(v) else f"{v:.4g}") if isinstance(v, (float, np.floating)) else str(v)

    def html_table(rs, cs):
        return ("<table border=1 cellpadding=3 style='border-collapse:collapse;font-size:12px'><tr>"
                + "".join(f"<th>{c}</th>" for c in cs) + "</tr>"
                + "".join("<tr>" + "".join(f"<td>{fmt(r[c])}</td>" for c in cs) + "</tr>" for r in rs) + "</table>")

    table = html_table(rows, cols)
    snr_table = html_table(snr_rows, list(snr_rows[0]))
    sensor_table = html_table([{"metric": k, "value": v} for k, v in sensor.items()], ["metric", "value"])
    html = ["<html><body style='font-family:sans-serif;max-width:1150px;margin:auto'>",
            f"<h1>NDR test: {meta['sensor']}{' (simulated)' if meta.get('fake') else ''}</h1>",
            f"<p>{meta['reads']} reads per reset, {meta['interval_ms']:g} ms apart; full well {meta['full_well']}; "
            f"ROI x {x0}-{x1}, y {y0}-{y1}. NDR settings: {meta.get('ndr_settings', {})}.</p>",
            "<p>min_increment_z is the smallest read-to-read rise in units of its noise: strongly negative means a "
            "read lost signal, which suggests it was destructive. In the dark the rise per read is tiny, so expect "
            "z near 0 there. slope_ratio compares NDR with destructive readout (expect ≈ 1). max_read_vs_ref "
            "compares each read with the reference capture at the same exposure.</p>", table,
            "<h2>SNR</h2>",
            "<p>Each estimator measures the same exposure, ending at the last read (integration_ms). "
            "Signal is the ROI mean. Noise is temporal: the per-pixel variance across repeats, pooled over the "
            "ROI, so fixed-pattern noise is excluded. spatial_spread_pct is the spatial spread of the per-pixel slope "
            "across the ROI: it equals fixed-pattern noise (PRNU/DSNU) only when the ROI is evenly lit. <b>ramp</b>: slope fitted through all reads × integration time. <b>cds</b>: last "
            "read − first read (covers less integration than the ramp). <b>single</b>: one destructive reference "
            "frame minus the dark reference, i.e. what a normal capture gives. ramp_vs_single > 1 is the SNR "
            "gained by NDR. Needs ≥ 2 repeats (and ≥ 2 reference frames for single); otherwise n/a.</p>",
            snr_table,
            "<h2>Sensor metrics</h2>",
            "<p>read_noise_adu: dark second-difference noise. conversion_gain from the bright CDS photon transfer, "
            "var = gain × mean + 2 × read_noise² (approximate: one signal level, assumes shot-noise-limited "
            "otherwise). dynamic_range_db = 20·log10((saturation − bias) / read noise).</p>", sensor_table]
    for state, ramp_png, map_png in sections:
        html += [f"<h2>{state}</h2>", f"<img src='data:image/png;base64,{ramp_png}'>",
                 f"<img src='data:image/png;base64,{map_png}'>"]
    (args.session / "report.html").write_text("".join(html) + "</body></html>")
    for r in rows:
        print(f"{r['light']:6s} rep {r['repeat']}: slope {r['ndr_slope_adu_per_ms']:.4g} ADU/ms "
              f"(ref {r['ref_slope_adu_per_ms']:.4g}), R² {r['r2']:.4f}, min inc z {r['min_increment_z']:.1f}, "
              f"noise {r['noise_adu']:.2f} ADU")
    for r in snr_rows:
        print(f"{r['light']:6s} SNR at {r['integration_ms']:g} ms: ramp {r['snr_ramp']:.4g}, CDS {r['snr_cds']:.4g}, "
              f"single {r['snr_single']:.4g} (ramp/single {r['ramp_vs_single']:.3g})")
    print("sensor: " + ", ".join(f"{k} {v:.4g}" for k, v in sensor.items()))
    print("report:", args.session / "report.html")


if __name__ == "__main__":
    main()
