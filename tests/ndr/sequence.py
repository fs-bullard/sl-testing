"""Capture one frame at each of a sequence of exposure times without resetting in between.

  python -m tests.ndr.sequence --sensor M5 --pvcfg C:\\path\\cfg.pvcfg --times 100 200 500 1000
  python -m tests.ndr.sequence --sensor test --fake

Writes len(--times) to register 48 first, so the sensor resets only after the final capture,
then captures one frame per exposure time and saves them all as a single TIFF stack
(sequence.tif, one page per time) plus meta.json. Register 48 is set back to 1 (normal readout)
at the end, even if the capture fails. With --fake the frames are independent simulated
captures (the fake does not model the non-reset).
"""
import argparse
import json
import time

import numpy as np
import tifffile

from sl_common import CameraDevice, output_dir
from sl_common.camera import resolve_interface
from sl_common.fake import FakeSLDevice
from sl_common.ui import ConsoleUI

from .ndr import NDR_OFF, NDR_REGISTER

SATURATION = 2**14 - 1


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--sensor", required=True, help="name used for the output folder")
    p.add_argument("--pvcfg", help="Pleora .pvcfg path (required on hardware)")
    p.add_argument("--times", type=float, nargs="+", default=[100, 200, 300, 400, 500],
                   help="exposure time (ms) of each capture, in order")
    p.add_argument("--full-well", choices=["High", "Low"], default="High")
    p.add_argument("--fake", action="store_true", help="simulated detector, no hardware")
    return p


def run(args, ui=None):
    """Capture the sequence; returns the session folder."""
    ui = ui or ConsoleUI()
    if not args.fake and not args.pvcfg:
        raise ValueError("--pvcfg is required for the GigE detector (or use --fake)")

    cam = CameraDevice(device=FakeSLDevice() if args.fake else None,
                       iface=None if args.fake else resolve_interface("gige"))
    out = output_dir(f"ndr_sequence/{args.sensor}")
    n = len(args.times)
    meta = {"sensor": args.sensor, "iface": "gige", "pvcfg": args.pvcfg, "full_well": args.full_well,
            "dds": False, "times_ms": args.times, "file": "sequence.tif", "fake": args.fake,
            "register_writes": [], "captures": []}

    def write_meta():
        (out / "meta.json").write_text(json.dumps(meta, indent=2))

    cam.open_camera(pvcfg_path=None if args.fake else args.pvcfg)
    frames = []
    try:
        cam.setup(dds=False)
        x, y = cam.dims()
        cam.register_write(NDR_REGISTER, n)  # reset only after the n-th capture
        meta["register_writes"].append({"register": NDR_REGISTER, "value": n})
        ui.log(f"register {NDR_REGISTER} = {n}")
        for i, t in enumerate(args.times):
            ui.check_stop()
            image = cam.acquire(x, y, 1, t, full_well=args.full_well)
            frame = image.Frame2Array(0).copy()
            frames.append(frame)
            meta["captures"].append({"index": i, "exposure_ms": t, "temp": cam.temperature(),
                                     "saturated_pixels": int((frame >= SATURATION).sum()), "time": time.time()})
            ui.log(f"{i + 1}/{n}  {t:g} ms  mean {frame.mean():8.1f}")
            ui.frame(np.stack(frames), f"sequence, {i + 1} of {n} captures")
            ui.progress(i + 1, n)
    finally:
        try:
            cam.register_write(NDR_REGISTER, NDR_OFF)
            meta["register_writes"].append({"register": NDR_REGISTER, "value": NDR_OFF})
        except Exception as e:
            ui.log(f"WARNING: failed to restore register {NDR_REGISTER} to {NDR_OFF}: {e}")
        cam.close_camera()
        if frames:
            # Explicit greyscale ImageJ stack: otherwise a 3-frame stack is guessed to be RGB
            tifffile.imwrite(out / "sequence.tif", np.stack(frames), imagej=True, photometric="minisblack",
                             metadata={"axes": "TYX"})
        write_meta()
    ui.log(f"saved {len(frames)} frames to {out / 'sequence.tif'}")
    return out


def main():
    p = build_parser()
    args = p.parse_args()
    try:
        run(args)
    except ValueError as e:
        p.error(str(e))


if __name__ == "__main__":
    main()
