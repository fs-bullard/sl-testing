"""Template for a new test. Copy this folder to tests/<name>/ and edit.

Run from the repo root:  python -m tests.<name>.run [--fake]
"""
import argparse

from sl_common import CameraDevice, output_dir
from sl_common.fake import FakeSLDevice

TEST_NAME = "template"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake", action="store_true", help="use the hardware-free fake device")
    args = parser.parse_args()

    cam = CameraDevice(device=FakeSLDevice() if args.fake else None)
    out = output_dir(TEST_NAME)
    cam.open_camera()
    try:
        image = cam.acquire(x_dim=64, y_dim=64, n_frames=2, exposure_ms=10)
        print("frame 0 mean:", image.Frame2Array(0).mean(), "->", out)
    finally:
        cam.close_camera()


if __name__ == "__main__":
    main()
