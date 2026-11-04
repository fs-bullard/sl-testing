"""Desktop GUI for the capture scripts.

  python capture_gui.py

Pick a test, fill in its settings and press Start. The settings form is built from the capture
script's own argparse parser (build_parser()), so new options appear here automatically, and
the capture runs that script's run(args, ui) in a worker thread. LED prompts appear as dialogs,
each capture is previewed as it arrives, and Stop takes effect between captures (an exposure
already in progress finishes first). "Analyse" runs the test's analysis script on the session
and opens its report.

Viewer: every capture of the session is kept (block-averaged to at most PREVIEW_MAX_PX, so
memory stays small) and can be scrolled with the slider or the Left/Right keys, frame by frame
or as the mean of its frames. Drag a rectangle on the image to set an ROI: the plots below show
its mean across all captures and across the frames of the current capture (z-profiles).
"Load session…" fills the viewer from a saved session folder, or any folder of TIFFs.

To add a test: give its capture module build_parser() and run(args, ui), then add it to CAPTURES.
"""
import argparse
import contextlib
import importlib
import io
import json
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import traceback
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
import tifffile

from sl_common.ui import Stopped

ROOT = Path(__file__).resolve().parent
SETTINGS = ROOT / "output" / "gui_settings.json"
# label -> (capture module, analysis module, output folder under output/)
CAPTURES = {
    "Blooming": ("tests.blooming.capture", "tests.blooming.analyze", "blooming"),
    "NDR (GigE)": ("tests.ndr.capture", "tests.ndr.analyze", "ndr"),
}
PREVIEW_MAX_PX = 600
SAT = 2**14 - 1


class GuiUI:
    """The `ui` passed to run(): called from the worker thread, so it only posts to the queue."""

    def __init__(self, q, stop):
        self.q, self.stop = q, stop

    def log(self, msg):
        self.q.put(("log", str(msg)))

    def prompt(self, msg):
        done = threading.Event()
        self.q.put(("prompt", msg, done))
        done.wait()
        self.check_stop()

    def frame(self, frames, label):
        frames = np.asarray(frames)
        frames = frames.reshape(-1, *frames.shape[-2:])
        small, scale = downsample(frames)
        stats = {"mean": float(small.mean()), "max": int(frames.max()), "saturated": int((frames >= SAT).sum()),
                 "shape": frames.shape}
        self.q.put(("frame", {"label": label, "frames": small, "scale": scale, "stats": stats}))

    def progress(self, done, total):
        self.q.put(("progress", done, total))

    def check_stop(self):
        if self.stop.is_set():
            raise Stopped("stopped by operator")


def downsample(frames):
    """Block-mean each frame by an integer factor so the longest side is <= PREVIEW_MAX_PX.
    Block means keep ROI means correct, unlike striding. Returns (float32 stack, factor)."""
    n, h, w = frames.shape
    f = max(1, int(np.ceil(max(h, w) / PREVIEW_MAX_PX)))
    h2, w2 = h // f * f, w // f * f
    small = [fr[:h2, :w2].astype(np.float32).reshape(h2 // f, f, w2 // f, f).mean((1, 3)) for fr in frames]
    return np.stack(small), f


def natural_key(path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def capture_label(entry):
    """Readable label from a meta.json capture entry (blooming or NDR)."""
    parts = [str(entry[k]) for k in ("kind", "light") if k in entry]
    if "exposure_ms" in entry:
        parts.append(f"{entry['exposure_ms']:g} ms")
    if "repeat" in entry:
        parts.append(f"rep {entry['repeat']}")
    return " ".join(parts) or entry["file"]


def load_session(folder, ui):
    """Feed a saved session (meta.json captures, else every TIFF in the folder) into the viewer."""
    meta = folder / "meta.json"
    if meta.exists():
        entries = [(folder / c["file"], capture_label(c)) for c in json.loads(meta.read_text())["captures"]]
    else:
        entries = [(p, p.stem) for p in sorted(folder.glob("*.tif*"), key=natural_key)]
    if not entries:
        raise ValueError(f"No captures found in {folder}")
    for i, (path, label) in enumerate(entries):
        ui.check_stop()
        frames = np.load(path) if path.suffix == ".npy" else tifffile.imread(path)
        ui.frame(frames, label)
        ui.progress(i + 1, len(entries))
    ui.log(f"loaded {len(entries)} captures from {folder}")
    return folder


def fmt(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


def split_args(text):
    """Split typed arguments like a shell would, keeping Windows backslash paths intact."""
    if os.name != "nt":
        return shlex.split(text)
    return [a[1:-1] if len(a) > 1 and a[0] == a[-1] == '"' else a for a in shlex.split(text, posix=False)]


def command_line(args):
    return subprocess.list2cmdline(args) if os.name == "nt" else shlex.join(args)


class Field:
    """One form widget for one argparse option, and its conversion back to argv."""

    def __init__(self, parent, row, action, module, saved):
        self.action = action
        self.opt = action.option_strings[-1]
        label = action.dest.replace("_", " ")
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
        cell = ttk.Frame(parent)
        cell.grid(row=row, column=1, sticky="ew", pady=2)
        default = saved if saved is not None else action.default

        if action.nargs == 0:  # store_true
            self.kind = "flag"
            self.var = tk.BooleanVar(value=bool(default))
            ttk.Checkbutton(cell, variable=self.var).pack(side="left")
        elif action.choices and action.nargs == "+":
            self.kind = "multi"
            chosen = set(default or [])
            self.vars = {c: tk.BooleanVar(value=c in chosen) for c in action.choices}
            for c, v in self.vars.items():
                ttk.Checkbutton(cell, text=c, variable=v).pack(side="left")
        elif action.choices:
            self.kind = "choice"
            self.var = tk.StringVar(value=fmt(default) if default is not None else "")
            ttk.Combobox(cell, textvariable=self.var, values=list(action.choices), state="readonly",
                         width=12).pack(side="left")
        else:
            self.kind = "list" if action.nargs == "+" else "text"
            if isinstance(default, (list, tuple)):
                default = " ".join(fmt(v) for v in default)
            self.var = tk.StringVar(value=fmt(default) if default is not None else "")
            known = list(getattr(module, "SENSOR_REGISTERS", {})) if action.dest == "sensor" else []
            if known:
                ttk.Combobox(cell, textvariable=self.var, values=known, width=28).pack(side="left", fill="x", expand=True)
            else:
                ttk.Entry(cell, textvariable=self.var, width=30).pack(side="left", fill="x", expand=True)
            if action.dest == "pvcfg":
                ttk.Button(cell, text="…", width=3, command=self.browse).pack(side="left", padx=(4, 0))
        if action.help:
            ttk.Label(parent, text=action.help, foreground="gray", wraplength=230).grid(
                row=row, column=2, sticky="w", padx=(8, 0))

    def browse(self):
        path = filedialog.askopenfilename(filetypes=[("Pleora config", "*.pvcfg"), ("All files", "*.*")])
        if path:
            self.var.set(path)

    def value(self):
        if self.kind == "multi":
            return [c for c, v in self.vars.items() if v.get()]
        return self.var.get()

    def argv(self):
        v = self.value()
        if self.kind == "flag":
            return [self.opt] if v else []
        if self.kind == "multi":
            return [self.opt, *v] if v else []
        v = v.strip()
        if not v:
            return []
        return [self.opt, *v.split()] if self.kind == "list" else [self.opt, v]


class App:
    def __init__(self, root):
        self.root = root
        self.q = queue.Queue()
        self.stop = threading.Event()
        self.worker = None
        self.last_out = None
        self.fields = []
        self.captures = []  # viewer history: dicts from GuiUI.frame
        self.idx = 0
        self.roi = None  # (x0, y0, x1, y1) in full-resolution pixels
        self.im = None
        try:
            self.settings = json.loads(SETTINGS.read_text())
        except (OSError, ValueError):
            self.settings = {}

        root.title("Better Better XView")
        root.geometry("1400x850")
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        paned = ttk.PanedWindow(root, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=8, pady=8)
        left, right = ttk.Frame(paned), ttk.Frame(paned)
        paned.add(left, weight=1)
        paned.add(right, weight=1)

        top = ttk.Frame(left)
        top.pack(fill="x")
        ttk.Label(top, text="Test").pack(side="left")
        self.test = tk.StringVar(value=self.settings.get("_test", next(iter(CAPTURES))))
        box = ttk.Combobox(top, textvariable=self.test, values=list(CAPTURES), state="readonly", width=20)
        box.pack(side="left", padx=6)
        box.bind("<<ComboboxSelected>>", lambda e: self.build_form())

        self.form = ttk.LabelFrame(left, text="Capture settings", padding=8)
        self.form.pack(fill="x", pady=8)

        ana = ttk.LabelFrame(left, text="Analysis", padding=8)
        ana.pack(fill="x")
        ttk.Label(ana, text="extra arguments").grid(row=0, column=0, sticky="w")
        self.ana_args = tk.StringVar()
        ttk.Entry(ana, textvariable=self.ana_args, width=40).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Label(ana, text="e.g. --pitch-um 74.8 for blooming", foreground="gray").grid(row=1, column=1, sticky="w", padx=6)

        buttons = ttk.Frame(left)
        buttons.pack(fill="x", pady=8)
        self.start_btn = ttk.Button(buttons, text="Start capture", command=self.start)
        self.stop_btn = ttk.Button(buttons, text="Stop", command=self.request_stop, state="disabled")
        self.ana_btn = ttk.Button(buttons, text="Analyse session…", command=self.analyse)
        self.open_btn = ttk.Button(buttons, text="Open folder", command=self.open_folder)
        self.load_btn = ttk.Button(buttons, text="Load session…", command=self.load)
        for i, b in enumerate((self.start_btn, self.stop_btn, self.load_btn, self.ana_btn, self.open_btn)):
            b.grid(row=i // 3, column=i % 3, sticky="ew", padx=(0, 6), pady=2)
        buttons.columnconfigure((0, 1, 2), weight=1)
        self.progress = ttk.Progressbar(left, mode="determinate")
        self.progress.pack(fill="x")
        self.status = tk.StringVar(value="Idle")
        ttk.Label(left, textvariable=self.status).pack(anchor="w", pady=(4, 0))

        logf = ttk.LabelFrame(left, text="Log", padding=4)
        logf.pack(fill="both", expand=True, pady=(8, 0))
        self.log_text = tk.Text(logf, height=12, wrap="none", font=("Consolas", 9))
        sb = ttk.Scrollbar(logf, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True)

        nav = ttk.Frame(right)
        nav.pack(fill="x")
        ttk.Button(nav, text="◀", width=3, command=lambda: self.step(-1)).pack(side="left")
        self.slider = tk.Scale(nav, orient="horizontal", from_=0, to=0, showvalue=False, command=self.on_slider)
        self.slider.pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(nav, text="▶", width=3, command=lambda: self.step(1)).pack(side="left")
        self.pos = tk.StringVar(value="0 / 0")
        ttk.Label(nav, textvariable=self.pos, width=9).pack(side="left", padx=4)
        ttk.Label(nav, text="frame").pack(side="left")
        self.frame_sel = tk.StringVar(value="mean")
        self.frame_box = ttk.Combobox(nav, textvariable=self.frame_sel, values=["mean"], state="readonly", width=6)
        self.frame_box.pack(side="left", padx=4)
        self.frame_box.bind("<<ComboboxSelected>>", lambda e: self.show_current())
        self.follow = tk.BooleanVar(value=True)
        ttk.Checkbutton(nav, text="follow new", variable=self.follow).pack(side="left", padx=(6, 0))

        self.fig = Figure(figsize=(6, 4.6))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.selector = None
        self.stats = tk.StringVar(value="No capture yet")
        ttk.Label(right, textvariable=self.stats, font=("Consolas", 10)).pack(anchor="w")

        roi_bar = ttk.Frame(right)
        roi_bar.pack(fill="x")
        self.roi_text = tk.StringVar(value="ROI: drag a rectangle on the image")
        ttk.Label(roi_bar, textvariable=self.roi_text, font=("Consolas", 10)).pack(side="left")
        ttk.Button(roi_bar, text="Save profile CSV…", command=self.save_profile).pack(side="right")
        ttk.Button(roi_bar, text="Clear ROI", command=self.clear_roi).pack(side="right", padx=4)
        self.zfig = Figure(figsize=(6, 2.4))
        self.zax_cap, self.zax_frame = self.zfig.subplots(1, 2)
        self.zcanvas = FigureCanvasTkAgg(self.zfig, master=right)
        self.zcanvas.get_tk_widget().pack(fill="x")
        self.update_profile()
        root.bind("<Left>", lambda e: self.key_step(-1))
        root.bind("<Right>", lambda e: self.key_step(1))

        self.build_form()
        self.root.after(50, self.poll)

    # ---- form
    def module(self):
        return importlib.import_module(CAPTURES[self.test.get()][0])

    def build_form(self):
        for w in self.form.winfo_children():
            w.destroy()
        mod = self.module()
        saved = self.settings.get(self.test.get(), {})
        actions = [a for a in mod.build_parser()._actions if not isinstance(a, argparse._HelpAction)]
        self.fields = [Field(self.form, i, a, mod, saved.get(a.dest)) for i, a in enumerate(actions)]
        self.form.columnconfigure(1, weight=1, minsize=190)
        self.ana_args.set(saved.get("_analysis_args", ""))

    def save_settings(self):
        self.settings["_test"] = self.test.get()
        self.settings[self.test.get()] = {f.action.dest: f.value() for f in self.fields}
        self.settings[self.test.get()]["_analysis_args"] = self.ana_args.get()
        try:
            SETTINGS.parent.mkdir(exist_ok=True)
            SETTINGS.write_text(json.dumps(self.settings, indent=2))
        except OSError:
            pass

    # ---- capture
    def start(self):
        mod_name = CAPTURES[self.test.get()][0]
        mod = self.module()
        argv = [a for f in self.fields for a in f.argv()]
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                args = mod.build_parser().parse_args(argv)
        except SystemExit:
            messagebox.showerror("Invalid settings", err.getvalue().strip().splitlines()[-1].split("error: ", 1)[-1])
            return
        self.save_settings()
        self.log(f"$ {command_line(['python', '-m', mod_name, *argv])}")
        self.launch(mod.run, args, f"Running {self.test.get()}…")

    def load(self):
        start = ROOT / "output" / CAPTURES[self.test.get()][2]
        path = filedialog.askdirectory(title="Session folder to view", initialdir=start if start.exists() else ROOT)
        if path:
            self.launch(load_session, Path(path), "Loading…")

    def launch(self, fn, arg, status):
        """Run fn(arg, ui) in the worker thread; the viewer starts a fresh history."""
        self.captures, self.idx, self.im = [], 0, None
        self.stop.clear()
        self.progress.configure(value=0, maximum=1)
        self.set_running(True, status)
        self.running_status = status
        self.worker = threading.Thread(target=self.work, args=(fn, arg), daemon=True)
        self.worker.start()

    def work(self, fn, arg):
        try:
            self.q.put(("done", fn(arg, GuiUI(self.q, self.stop))))
        except Stopped as e:
            self.q.put(("stopped", str(e)))
        except Exception:
            self.q.put(("error", traceback.format_exc()))

    def request_stop(self):
        self.stop.set()
        self.status.set("Stopping after the current capture…")

    def set_running(self, running, status):
        self.start_btn.configure(state="disabled" if running else "normal")
        self.ana_btn.configure(state="disabled" if running else "normal")
        self.load_btn.configure(state="disabled" if running else "normal")
        self.stop_btn.configure(state="normal" if running else "disabled")
        self.status.set(status)

    def poll(self):
        try:
            while True:
                self.handle(*self.q.get_nowait())
        except queue.Empty:
            pass
        self.root.after(50, self.poll)

    def handle(self, kind, *data):
        if kind == "log":
            self.log(data[0])
        elif kind == "prompt":
            msg, done = data
            self.status.set("Waiting for operator")
            if not messagebox.askokcancel("Operator action", f"{msg}\n\nOK when done. Cancel stops the capture."):
                self.stop.set()
            self.status.set(self.running_status)
            done.set()
        elif kind == "frame":
            self.add_capture(data[0])
        elif kind == "progress":
            done, total = data
            self.progress.configure(value=done, maximum=max(total, 1))
            self.status.set(f"{self.running_status} {done}/{total}")
        elif kind == "done":
            self.last_out = data[0]
            self.set_running(False, f"Finished: {data[0]}")
        elif kind == "stopped":
            self.log("Stopped; anything already captured was kept and the camera (if open) was closed.")
            self.set_running(False, "Stopped")
        elif kind == "error":
            self.log(data[0])
            self.set_running(False, "Failed, see log")
            messagebox.showerror("Failed", data[0].strip().splitlines()[-1])
        elif kind == "analysis_done":
            self.ana_btn.configure(state="normal")
            report = Path(data[0]) / "report.html"
            if data[1] == 0 and report.exists():
                self.status.set(f"Analysis done: {report}")
                webbrowser.open(report.as_uri())
            else:
                self.status.set("Analysis failed, see log")

    # ---- viewer
    def add_capture(self, cap):
        self.captures.append(cap)
        n = len(self.captures)
        self.slider.configure(to=n - 1)
        if self.follow.get() or n == 1:
            self.idx = n - 1
            self.slider.set(self.idx)
            self.show_current()
        else:
            self.pos.set(f"{self.idx + 1} / {n}")
            self.update_profile()

    def on_slider(self, value):
        idx = int(float(value))
        if self.captures and idx != self.idx:
            self.idx = idx
            self.show_current()

    def step(self, delta):
        if self.captures:
            self.idx = min(max(self.idx + delta, 0), len(self.captures) - 1)
            self.slider.set(self.idx)
            self.show_current()

    def key_step(self, delta):
        if not isinstance(self.root.focus_get(), (tk.Entry, ttk.Entry, tk.Text)):
            self.step(delta)

    def current_image(self, cap=None, frame=None):
        cap = cap or self.captures[self.idx]
        sel = self.frame_sel.get() if frame is None else frame
        return cap["frames"].mean(0) if sel == "mean" else cap["frames"][int(sel) - 1]

    def show_current(self):
        cap = self.captures[self.idx]
        n_frames = len(cap["frames"])
        self.frame_box.configure(values=["mean"] + [str(i + 1) for i in range(n_frames)])
        if self.frame_sel.get() != "mean" and int(self.frame_sel.get()) > n_frames:
            self.frame_sel.set("mean")
        img = self.current_image(cap)
        f = cap["scale"]
        extent = (0, img.shape[1] * f, img.shape[0] * f, 0)  # axes in full-resolution pixels
        if self.im is None or self.im.get_array().shape != img.shape:
            self.ax.clear()
            self.im = self.ax.imshow(img, cmap="gray", extent=extent, interpolation="nearest")
            self.ax.set_axis_off()
            self.make_selector()
        else:
            self.im.set_data(img)
            self.im.set_extent(extent)
        lo, hi = np.percentile(img, [1, 99.5])
        self.im.set_clim(lo, max(hi, lo + 1))
        frame_txt = "mean of frames" if self.frame_sel.get() == "mean" else f"frame {self.frame_sel.get()}"
        self.ax.set_title(f"{cap['label']}  ({frame_txt})", fontsize=10)
        self.canvas.draw_idle()
        st = cap["stats"]
        self.pos.set(f"{self.idx + 1} / {len(self.captures)}")
        self.stats.set(f"{st['shape'][2]}×{st['shape'][1]} px, {st['shape'][0]} frame(s)   mean {st['mean']:.1f}   "
                       f"max {st['max']}   saturated px {st['saturated']}")
        self.update_profile()

    def make_selector(self):
        self.selector = RectangleSelector(self.ax, self.on_select, useblit=False, interactive=True, button=[1],
                                          minspanx=2, minspany=2, spancoords="data",
                                          props={"edgecolor": "red", "fill": False, "linewidth": 1.5})
        if self.roi:
            x0, y0, x1, y1 = self.roi
            self.selector.extents = (x0, x1, y0, y1)

    def on_select(self, press, release):
        if None in (press.xdata, release.xdata, press.ydata, release.ydata):
            return
        x0, x1 = sorted((press.xdata, release.xdata))
        y0, y1 = sorted((press.ydata, release.ydata))
        self.roi = (int(x0), int(y0), int(np.ceil(x1)), int(np.ceil(y1)))
        self.update_profile()

    def clear_roi(self):
        self.roi = None
        if self.selector:
            self.selector.set_visible(False)
            self.selector.extents = (0, 0, 0, 0)
            self.canvas.draw_idle()
        self.update_profile()

    def roi_values(self, img, scale):
        x0, y0, x1, y1 = self.roi
        r0, c0 = int(y0 // scale), int(x0 // scale)
        return img[r0:max(int(np.ceil(y1 / scale)), r0 + 1), c0:max(int(np.ceil(x1 / scale)), c0 + 1)]

    def profiles(self):
        """ROI mean per capture (mean of its frames) and per frame of the current capture."""
        per_cap = [float(self.roi_values(self.current_image(c, "mean"), c["scale"]).mean()) for c in self.captures]
        cur = self.captures[self.idx]
        per_frame = [float(self.roi_values(fr, cur["scale"]).mean()) for fr in cur["frames"]]
        return per_cap, per_frame

    def update_profile(self):
        for a in (self.zax_cap, self.zax_frame):
            a.clear()
        if not self.captures or not self.roi:
            self.zax_cap.text(0.5, 0.5, "drag a rectangle on the image\nto plot its z-profile",
                              ha="center", va="center", transform=self.zax_cap.transAxes, color="gray")
            self.zax_frame.set_axis_off()
            self.zax_cap.set_xticks([]); self.zax_cap.set_yticks([])
            self.roi_text.set("ROI: drag a rectangle on the image")
            self.zcanvas.draw_idle()
            return
        self.zax_frame.set_axis_on()
        per_cap, per_frame = self.profiles()
        cur = self.captures[self.idx]
        vals = self.roi_values(self.current_image(), cur["scale"])
        x0, y0, x1, y1 = self.roi
        self.roi_text.set(f"ROI x {x0}-{x1}, y {y0}-{y1} ({x1 - x0}×{y1 - y0} px)   "
                          f"mean {vals.mean():.1f}   std {vals.std():.1f}")
        self.zax_cap.plot(range(1, len(per_cap) + 1), per_cap, "o-", ms=3)
        self.zax_cap.plot([self.idx + 1], [per_cap[self.idx]], "o", c="red", ms=6)
        if len(per_cap) <= 30:
            self.zax_cap.set_xticks(range(1, len(per_cap) + 1))
            self.zax_cap.set_xticklabels([c["label"] for c in self.captures], rotation=60, ha="right", fontsize=6)
        self.zax_cap.set_title("ROI mean across captures", fontsize=9)
        self.zax_frame.plot(range(1, len(per_frame) + 1), per_frame, "o-", ms=3, c="tab:green")
        if self.frame_sel.get() != "mean":
            k = int(self.frame_sel.get())
            self.zax_frame.plot([k], [per_frame[k - 1]], "o", c="red", ms=6)
        self.zax_frame.set_xlabel("frame", fontsize=8)
        self.zax_frame.set_title("ROI mean across frames of this capture", fontsize=9)
        for a in (self.zax_cap, self.zax_frame):
            a.tick_params(labelsize=7)
            a.set_ylabel("ADU", fontsize=8)
        self.zfig.tight_layout()
        self.zcanvas.draw_idle()

    def save_profile(self):
        if not self.captures or not self.roi:
            messagebox.showinfo("No profile", "Select an ROI on the image first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                            initialdir=self.last_out or ROOT / "output")
        if not path:
            return
        x0, y0, x1, y1 = self.roi
        with open(path, "w") as f:
            f.write(f"# ROI x {x0}-{x1}, y {y0}-{y1} (full-resolution px; values from block-averaged previews)\n")
            f.write("capture,label,frame,roi_mean_adu\n")
            for i, c in enumerate(self.captures):
                f.write(f"{i + 1},{c['label']},mean,{self.roi_values(c['frames'].mean(0), c['scale']).mean():.3f}\n")
                for k, fr in enumerate(c["frames"]):
                    f.write(f"{i + 1},{c['label']},{k + 1},{self.roi_values(fr, c['scale']).mean():.3f}\n")
        self.log(f"saved ROI profile to {path}")

    def log(self, msg):
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")

    # ---- analysis / files
    def session_folder(self):
        if self.last_out:
            return Path(self.last_out)
        start = ROOT / "output" / CAPTURES[self.test.get()][2]
        path = filedialog.askdirectory(title="Session folder", initialdir=start if start.exists() else ROOT)
        return Path(path) if path else None

    def analyse(self):
        session = self.session_folder()
        if not session:
            return
        self.save_settings()
        cmd = [sys.executable, "-m", CAPTURES[self.test.get()][1], str(session), *split_args(self.ana_args.get())]
        self.log(f"$ {command_line(cmd)}")
        self.ana_btn.configure(state="disabled")
        self.status.set("Analysing…")

        def run():
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                self.q.put(("log", line.rstrip()))
            self.q.put(("analysis_done", session, proc.wait()))

        threading.Thread(target=run, daemon=True).start()

    def open_folder(self):
        path = Path(self.last_out) if self.last_out else ROOT / "output"
        if os.name == "nt":
            os.startfile(path)
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])

    def on_close(self):
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("Capture running", "Press Stop and wait for the capture to finish, so the "
                                                      "camera is closed and the LED/NDR state restored.")
            return
        self.save_settings()
        self.root.destroy()


def main():
    if os.name == "nt":  # sharp text on high-DPI Windows displays instead of bitmap-scaled blur
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
