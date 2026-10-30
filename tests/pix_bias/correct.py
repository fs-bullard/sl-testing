"""Average and dark-correct the pixel-bias (10 uA vs 20 uA) captures.

  python -m tests.pix_bias.correct [--data data/pix_bias] [--out output/pix_bias]

Expects <data>/<N>ms/{N}ms_pbc_<bias>uA.tif and dark_{N}ms_pbc_<bias>uA.tif (multi-frame uint16).
Writes <out>/{N}ms_pbc_<bias>uA_dc.tif as float32: mean(bright) - mean(dark) + offset, clipped to 14 bits.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import tifffile

SAT = 2**14 - 1


def correct(bright_path, dark_path, offset):
    bright = tifffile.imread(bright_path).astype(np.float32)
    dark = tifffile.imread(dark_path).astype(np.float32)
    if bright.shape[-2:] != dark.shape[-2:]:
        raise ValueError(f"{bright_path.name} and {dark_path.name} differ in size")
    return np.clip(bright.reshape(-1, *bright.shape[-2:]).mean(0)
                   - dark.reshape(-1, *dark.shape[-2:]).mean(0) + offset, 0, SAT)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data/pix_bias"))
    p.add_argument("--out", type=Path, default=Path("output/pix_bias"))
    p.add_argument("--offset", type=float, default=100)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r"^(\d+ms_pbc_\d+uA)\.tif$")
    for bright in sorted(args.data.glob("*ms/*.tif"), key=lambda f: (int(f.parent.name[:-2]), f.name)):
        m = pattern.match(bright.name)
        if not m:
            continue
        dark = bright.with_name(f"dark_{bright.name}")
        if not dark.exists():
            print(f"skip {bright.name}: no {dark.name}")
            continue
        out = args.out / f"{m.group(1)}_dc.tif"
        tifffile.imwrite(out, correct(bright, dark, args.offset))
        print(f"{bright.name} -> {out}")


if __name__ == "__main__":
    main()
