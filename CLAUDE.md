# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

A collection of unrelated test scripts for detectors driven through `SLDevicePythonWrapper`. See [SLDevicePythonWrapper.md](SLDevicePythonWrapper.md) for the API reference. Read it before writing code.

- `sl_common/` is the shared package: `CameraDevice` (error-checked wrapper, `camera.py`), `FakeSLDevice` (hardware-free fake, `fake.py`), `output_dir()` (`output.py`), and the guarded SDK import (`sdk.py`).
- `tests/<name>/` holds one folder per test, with `run.py` as the entry point. Start new tests by copying `tests/_template/`.
- `output/` holds captured images and logs, written to `output/<test_name>/<timestamp>/`. It is gitignored.
- The SDK `.pyd` and DLLs are not in the repo. They must be on the Python path.

## Commands

Run from the repo root (tests import `sl_common`, so use `-m`):

```
pip install -r requirements.txt
python -m tests.<name>.run          # real hardware (Windows)
python -m tests.<name>.run --fake   # FakeSLDevice, works on macOS
```

Blooming test (`tests/blooming/`): `capture` takes darks (LED off) then brights (LED on) per exposure, low full well, no DDS, 3 frames per trigger. `analyze` corrects (bright mean − dark mean + 100), places ROIs, writes `metrics.csv` and `report.html` (per-exposure image plus a surface plot of each ROI).

```
python -m tests.blooming.capture --sensor M4 --iface usb      # --iface gige --pvcfg <file> for GigE; --fake to dry-run
python -m tests.blooming.analyze output/blooming/M4/<timestamp> --pitch-um <um>
```

`rois.json` in the session folder is created on first analysis and reused (edit it to move ROIs; `--regen-rois` rebuilds).

New test scripts should accept `--fake` and build `CameraDevice(device=FakeSLDevice())` when it is set. Extend `FakeSLDevice` as tests need more of the API. There is no linter or test runner configured.

## Environment constraints

- `SLDevicePythonWrapper` is a compiled pybind11 `.pyd`. It is Windows-only and locked to a specific CPython version. It is not pip-installable and has no source, stubs or type hints.
- `SLDeviceLib.dll`, `SLImage.dll`, `SLDefectCorrection.dll`, `SLHelperLibrary.dll` and the MSVC runtime DLLs must sit beside the `.pyd`. Bundle all of them if packaging with PyInstaller.
- This machine is macOS (Darwin), so the module cannot be imported here. Code that touches it can't be run locally. Use a fake device implementing the same surface for tests.
- Signatures of less-common methods are unverified. Check `help(SLDevice.MethodName)` at runtime before relying on arity or keyword names.

## Key conventions

- **Errors are return values, not exceptions.** Most methods return an `SLError`. Check every call against `SLError.SL_ERROR_SUCCESS`. The intended design is a thin `CameraDevice` wrapper with a `_check(err, op)` helper that raises `CameraNotConnectedError` (for `SL_ERROR_NOT_FOUND`) or `CameraError`. Call sites should go through the wrapper rather than the raw `SLDevice`.
- Some getters return a tuple `(SLError, value)`, such as `GetExposureTime`, `GetExposureMode` and `GetTestMode`. `AcquireImage` returns an `SLBufferInfo` whose error is on `.error`.
- The exposure getter reports 10 µs units, not the ms that was set.
- Configure first, then stream. Call `SetExposureTime`, `SetNumberOfFrames` and `SetFullWell` before `StartStream()`. Always pair `StartStream()` with `StopStream()` in `try/finally`. Once after `OpenCamera()`, call `SetExposureMode(ExposureModes.seq_mode)` and `SetDDS(False)`.
- `SLImage` is a stack of frames. `Frame2Array(0)` returns only the first frame.
- Data is 14-bit in a uint16 container (max 16383). Invert as `2**14 - arr` and clip corrections to `[0, 2**14 - 1]`.
- Corrections (`OffsetCorrection`, `GainCorrection`, `DefectCorrection`) require matching dimensions, otherwise `SL_ERROR_CORRECTION`. Rotating by 90 or 270 degrees swaps width and height, so don't rotate in place.
- `SetTestMode(True)` gives firmware-generated test images without a sensor.
- `SL_ERROR_ABROTED` is misspelled in the library itself. Use that spelling.
