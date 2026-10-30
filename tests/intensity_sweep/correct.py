"""Average and dark-correct the source-intensity sweep.

  python -m tests.intensity_sweep.correct [--data data/intensity_sweep] [--out output/intensity_sweep]

Each sub-folder of <data> is one sweep: <P>_pc_<rest>.tif plus a matching dark_<rest>.tif
(e.g. 45_pc_500_ms.tif / dark_500_ms.tif, 10_pc_1000ms_20_uA.tif / dark_1000ms_20_uA.tif).
Any number of frames per file. Writes <out>/<folder>/<P>pc_dc.tif as float32:
mean(bright) - mean(dark) + offset, clipped to 14 bits.
"""
import argparse
import re
from pathlib import Path

import tifffile

from tests.pix_bias.correct import correct


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data/intensity_sweep"))
    p.add_argument("--out", type=Path, default=Path("output/intensity_sweep"))
    p.add_argument("--offset", type=float, default=100)
    args = p.parse_args()

    pattern = re.compile(r"^(\d+)_pc_(.+)\.tif$")
    found = [(m, f) for f in args.data.glob("*/*.tif") if (m := pattern.match(f.name))]
    for m, bright in sorted(found, key=lambda mf: (mf[1].parent.name, int(mf[0].group(1)))):
        pc, rest = m.groups()
        dark = bright.with_name(f"dark_{rest}.tif")
        if not dark.exists():
            print(f"skip {bright.name}: no {dark.name}")
            continue
        out_dir = args.out / bright.parent.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{pc}pc_dc.tif"
        tifffile.imwrite(out, correct(bright, dark, args.offset))
        print(f"{bright.name} -> {out}")


if __name__ == "__main__":
    main()
