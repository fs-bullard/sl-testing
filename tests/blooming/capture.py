"""Blooming capture: darks (LED off) then brights (LED on) at each exposure time.

Run from the repo root:
  python -m tests.blooming.capture --sensor M4 --iface usb
  python -m tests.blooming.capture --sensor M5 --iface gige --pvcfg C:\\path\\cfg.pvcfg
  python -m tests.blooming.capture --sensor test --fake
Settings: low full well, no DDS, 3 frames per trigger (averaged in analysis).
"""
import argparse
import json
import time

import numpy as np
import tifffile

from sl_common import CameraDevice, output_dir
from sl_common.camera import resolve_interface
from sl_common.fake import FakeSLDevice

EXPOSURES_MS = [100, 200, 500, 750, 1000, 2000, 3000, 4000, 5000, 7000, 10000]
SATURATION = 2**14 - 1
# Upper-cased --sensor -> {address: value}, written after setup. M5 / TRITON still to be filled in.
SENSOR_REGISTERS = {
    "M4": {381: 33000},
    "M5": {},
    "TRITON": {},
}


def set_led(cam, on):
    if hasattr(cam.device, "led_on"):
        cam.device.led_on = on
    else:
        input(f"Turn the LED {'ON' if on else 'OFF'} (box stays in place), then press Enter... ")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--sensor", required=True, help="name used for the output folder")
    p.add_argument("--iface", choices=["usb", "gige"], default="usb")
    p.add_argument("--pvcfg", help="Pleora .pvcfg path (GigE only)")
    p.add_argument("--exposures", type=float, nargs="+", default=EXPOSURES_MS)
    p.add_argument("--frames", type=int, default=3)
    p.add_argument("--fake", action="store_true")
    args = p.parse_args()

    cam = CameraDevice(device=FakeSLDevice() if args.fake else None,
                       iface=None if args.fake else resolve_interface(args.iface))
    out = output_dir(f"blooming/{args.sensor}")
    meta = {"sensor": args.sensor, "iface": args.iface, "full_well": "Low", "dds": False,
            "frames_per_trigger": args.frames, "captures": []}

    def capture(kind, exp):
        x, y = cam.dims()
        t0 = cam.temperature()
        image = cam.acquire(x, y, args.frames, exp, full_well="Low")
        frames = np.stack([image.Frame2Array(i).copy() for i in range(args.frames)])
        name = f"{kind}_{exp:g}ms.tif"
        # Explicit greyscale ImageJ stack: otherwise a 3-frame stack is guessed to be RGB
        tifffile.imwrite(out / name, frames, imagej=True, photometric="minisblack",
                         metadata={"axes": "TYX"})
        meta["captures"].append({
            "kind": kind, "exposure_ms": exp, "file": name, "temp_before": t0,
            "temp_after": cam.temperature(), "saturated_pixels": int((frames >= SATURATION).sum()),
            "time": time.time()})
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"{kind:6s} {exp:7g} ms  mean {frames.mean():8.1f}  std {frames.std():7.1f}  "
              f"T {t0} -> {meta['captures'][-1]['temp_after']}  "
              f"saturated px {meta['captures'][-1]['saturated_pixels']}")

    cam.open_camera(pvcfg_path=args.pvcfg)
    try:
        cam.setup(dds=False)
        for addr, value in SENSOR_REGISTERS.get(args.sensor.upper(), {}).items():
            cam.register_write(addr, value)
            meta.setdefault("registers", {})[addr] = value
        set_led(cam, False)
        for exp in args.exposures:
            capture("dark", exp)
        set_led(cam, True)
        try:
            for exp in args.exposures:
                capture("bright", exp)
        finally:
            set_led(cam, False)
    finally:
        cam.close_camera()
    print("saved to", out)


if __name__ == "__main__":
    main()
