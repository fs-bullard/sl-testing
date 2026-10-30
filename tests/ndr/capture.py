"""Non-destructive readout (NDR) capture over GigE.

  python -m tests.ndr.capture --sensor M5 --pvcfg C:\\path\\cfg.pvcfg
  python -m tests.ndr.capture --sensor test --fake

For each light state (dark, then bright):
  1. Reference: normal destructive captures at each NDR read time, for comparison.
  2. NDR: --repeats integrations, each one reset followed by --reads non-destructive reads.
The NDR calls themselves are placeholders in tests/ndr/ndr.py (search TODO(NDR)); with --fake
they are simulated. Analyse with: python -m tests.ndr.analyze <session folder>
"""
import argparse
import json
import time

import numpy as np
import tifffile

from sl_common import CameraDevice, output_dir
from sl_common.camera import resolve_interface
from sl_common.fake import FakeSLDevice

from .ndr import FakeNdr, SdkNdr, nominal_read_times

SATURATION = 2**14 - 1


def set_led(cam, on):
    if hasattr(cam.device, "led_on"):
        cam.device.led_on = on
    else:
        input(f"Turn the LED {'ON' if on else 'OFF'}, then press Enter... ")


def save_stack(path, frames):
    # Explicit greyscale ImageJ stack: otherwise a 3-frame stack is guessed to be RGB
    tifffile.imwrite(path, frames, imagej=True, photometric="minisblack", metadata={"axes": "TYX"})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--sensor", required=True, help="name used for the output folder")
    p.add_argument("--pvcfg", help="Pleora .pvcfg path (required on hardware)")
    p.add_argument("--reads", type=int, default=10, help="NDR reads per reset")
    p.add_argument("--interval", type=float, default=100, help="ms between NDR reads")
    p.add_argument("--repeats", type=int, default=3, help="NDR integrations per light state")
    p.add_argument("--ref-frames", type=int, default=3, help="frames per destructive reference capture")
    p.add_argument("--no-reference", action="store_true", help="skip the destructive reference captures")
    p.add_argument("--light", nargs="+", choices=["dark", "bright"], default=["dark", "bright"])
    p.add_argument("--full-well", choices=["High", "Low"], default="High")
    p.add_argument("--fake", action="store_true")
    args = p.parse_args()
    if not args.fake and not args.pvcfg:
        p.error("--pvcfg is required for the GigE detector (or use --fake)")

    cam = CameraDevice(device=FakeSLDevice() if args.fake else None,
                       iface=None if args.fake else resolve_interface("gige"))
    ndr = FakeNdr(cam) if args.fake else SdkNdr(cam)
    out = output_dir(f"ndr/{args.sensor}")
    read_times = nominal_read_times(args.reads, args.interval)
    meta = {"sensor": args.sensor, "iface": "gige", "pvcfg": args.pvcfg, "full_well": args.full_well,
            "dds": False, "reads": args.reads, "interval_ms": args.interval, "nominal_read_times_ms": read_times,
            "fake": args.fake, "captures": []}

    def log(entry):
        meta["captures"].append({**entry, "temp": cam.temperature(), "time": time.time()})
        (out / "meta.json").write_text(json.dumps(meta, indent=2))

    cam.open_camera(pvcfg_path=None if args.fake else args.pvcfg)
    try:
        cam.setup(dds=False)
        x, y = cam.dims()
        for state in args.light:
            set_led(cam, state == "bright")
            if not args.no_reference:
                for t in read_times:
                    image = cam.acquire(x, y, args.ref_frames, t, full_well=args.full_well)
                    frames = np.stack([image.Frame2Array(i).copy() for i in range(args.ref_frames)])
                    name = f"ref_{state}_{t:g}ms.tif"
                    save_stack(out / name, frames)
                    log({"kind": "reference", "light": state, "exposure_ms": t, "file": name,
                         "saturated_pixels": int((frames >= SATURATION).sum())})
                    print(f"ref  {state:6s} {t:7g} ms  mean {frames.mean():8.1f}")

            ndr.configure(args.reads, args.interval)
            try:
                for rep in range(args.repeats):
                    reads, times = ndr.acquire(args.reads, args.interval)
                    name = f"ndr_{state}_rep{rep}.tif"
                    save_stack(out / name, reads)
                    log({"kind": "ndr", "light": state, "repeat": rep, "file": name, "read_times_ms": list(times),
                         "saturated_pixels": int((reads >= SATURATION).sum())})
                    means = reads.reshape(len(reads), -1).mean(1)
                    print(f"ndr  {state:6s} rep {rep}  read means {means[0]:.1f} -> {means[-1]:.1f}")
            finally:
                ndr.disable()
            meta["ndr_settings"] = ndr.settings
        set_led(cam, False)
    finally:
        cam.close_camera()
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("saved to", out)


if __name__ == "__main__":
    main()
