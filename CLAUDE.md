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
python -m tests.<name>.run --fake   # FakeSLDevice, no SDK or hardware needed
```

Blooming test (`tests/blooming/`): `capture` takes darks (LED off) then brights (LED on) per exposure, low full well, no DDS, 3 frames per trigger. Per-sensor register writes (`SENSOR_REGISTERS` in `capture.py`, keyed by upper-cased `--sensor`) run after setup and are logged in `meta.json`. `analyze` corrects (bright mean − dark mean + 100), places ROIs, writes `metrics.csv` and `report.html` (per-exposure image plus a surface plot of each ROI).

```
python -m tests.blooming.capture --sensor M4 --iface usb      # --iface gige --pvcfg <file> for GigE; --fake to dry-run
python -m tests.blooming.analyze output/blooming/M4/<timestamp> --pitch-um <um>
```

On real hardware `capture` pauses with an `input()` prompt for the operator to switch the LED. With `--fake` it sets `FakeSLDevice.led_on` instead. Each capture is saved as a multi-page uint16 TIFF frame stack (`analyze` still reads older `.npy` sessions), and `meta.json` logs exposure, temperatures and saturated-pixel counts.

ROIs are equal-area `--roi-long-mm` × `--roi-short-mm` rectangles `--roi-dist-mm` from the core, long side tangential (top/bottom wide, left/right tall), plus a wide control in the farthest corner. `rois.json` in the session folder is created on first analysis and reused (edit it to move ROIs; `--regen-rois` rebuilds).

Pixel-bias data (`data/pix_bias/<N>ms/`, 3-frame uint16 TIFFs for 10/20 uA with matching `dark_` files): `python -m tests.pix_bias.correct` writes mean(bright) − mean(dark) + 100 as float32 `<N>ms_pbc_<bias>uA_dc.tif` to `output/pix_bias/`.

`python -m tests.pix_bias.report` then compares 10 vs 20 uA: row-band profiles left and right of the saturated spot (log scale, plus 20−10 difference) per exposure, written to `output/pix_bias/report.html` and `profiles.csv`.

Intensity sweeps (`data/intensity_sweep/<sweep>/`, files `<P>_pc_<rest>.tif` with a matching `dark_<rest>.tif`, any frame count): `python -m tests.intensity_sweep.correct` writes `output/intensity_sweep/<sweep>/<P>pc_dc.tif` (reuses `tests.pix_bias.correct.correct`). `python -m tests.intensity_sweep.analyze <sweep>` reports two contrasts against intensity and spot overdrive: the inner box against its surround, and the light-box footprint against just outside it. Overdrive for saturated images is P(0)/P(R), using the radial profile P of the brightest unsaturated image and the saturated-disc radius R, so it doesn't assume a linear LED. Region rectangles are per sweep in `REGIONS` (CLI-overridable); add an entry for each new sweep and check it on the report's overview image.

NDR test (`tests/ndr/`, GigE only): `capture` records destructive reference captures at each read time, then `--repeats` NDR integrations (one reset, `--reads` reads `--interval` ms apart) per light state. NDR is switched by register 48: write 0 to enable and 1 to disable (`NDR_REGISTER` in `tests/ndr/ndr.py`). `capture` disables it at start-up and after each light state. `SdkNdr.acquire` assumes the reads arrive as one triggered frame sequence at `--interval` ms per frame. This is unverified (`TODO(NDR)`). `FakeNdr` makes the same register writes and simulates the ramp from `FakeSLDevice.signal()`. `analyze` checks that reads are non-destructive (read-to-read increments), linearity, NDR slope against the destructive reference, and noise (second differences / √6). It also writes per-pixel slope maps, using only reads before a pixel saturates. `snr.csv` compares the SNR of three estimators at the same integration time: the ramp fit, CDS (last read − first read), and one destructive reference frame. Each pools per-pixel variance across repeats, so it needs at least 2 repeats. `sensor_metrics.csv` gives read noise, conversion gain (from bright CDS photon transfer), dark current, bias and dynamic range.

```
python -m tests.ndr.capture --sensor M5     # --fake to dry-run; --pvcfg <file> optional
python -m tests.ndr.analyze output/ndr/M5/<timestamp>
```

`python -m tests.ndr.sequence --sensor M5 --times 100 200 500` writes len(times) to register 48 so the sensor resets only after the last capture. It then captures one frame per exposure time and saves them as a single `sequence.tif` stack in `output/ndr_sequence/<sensor>/<timestamp>/`. Register 48 is restored to 1 in `finally`. The script has no analysis step.

GUI: `python capture_gui.py` (tkinter) runs any capture listed in its `CAPTURES` dict. It builds the settings form from the module's `build_parser()` and calls `run(args, ui)` in a worker thread. Capture scripts must therefore expose those two functions. They must also talk to the operator only through `ui` (`sl_common/ui.py`: `log`, `prompt`, `frame`, `progress`, `check_stop`), never `print`/`input`. `ConsoleUI` is the command-line default. Stop takes effect at the next `check_stop()`, which raises `Stopped`, so cleanup belongs in `finally`. The GUI's last-used settings are saved in `output/gui_settings.json`. The viewer keeps every capture of the session as a block-averaged stack (`downsample()`, at most `PREVIEW_MAX_PX`; block means keep ROI means exact). An ROI dragged on the image plots its mean across captures and across frames. "Load session…" fills the viewer from `meta.json`, or from any folder of TIFFs.

New test scripts should accept `--fake` and build `CameraDevice(device=FakeSLDevice())` when it is set. Extend `FakeSLDevice` as tests need more of the API.

How the fake plugs in: `CameraDevice._sdk()` returns `device.sdk` when the device has `is_fake`, otherwise the real module from `sl_common.sdk.require_sdk()`. So any SDK type `CameraDevice` touches (`SLImage`, `FullWellModes`, `ExposureModes`, ...) must also be added to `FakeSLDevice.sdk`. `_check` compares `err.name` rather than identity, so the fake's own `SLError` enum works. Getters go through `_value()`, which accepts a `(SLError, value)` tuple or a bare value. There is no linter or test runner configured.

## Environment constraints

- `SLDevicePythonWrapper` is a compiled pybind11 `.pyd`. It is Windows-only and locked to a specific CPython version. It is not pip-installable and has no source, stubs or type hints.
- `SLDeviceLib.dll`, `SLImage.dll`, `SLDefectCorrection.dll`, `SLHelperLibrary.dll` and the MSVC runtime DLLs must sit beside the `.pyd`. Bundle all of them if packaging with PyInstaller.
- This machine is Windows with the SDK installed (SDK 2.2.19.0 per its log), so real-hardware runs are possible here. `--fake` still works without the SDK, for example on macOS.
- The USB SDK writes `SLSDK_USB_log.txt` to the current directory. Check it when a call fails without explanation.
- Signatures of less-common methods are unverified. Check `help(SLDevice.MethodName)` at runtime before relying on arity or keyword names.

## Key conventions

- **Errors are return values, not exceptions.** Most methods return an `SLError`. Check every call against `SLError.SL_ERROR_SUCCESS`. `CameraDevice` (`sl_common/camera.py`) does this with a `_check(err, op)` helper that raises `CameraNotConnectedError` (for `SL_ERROR_NOT_FOUND`) or `CameraError`. Call sites should go through the wrapper rather than the raw `SLDevice`.
- Some getters return a tuple `(SLError, value)`, such as `GetExposureTime`, `GetExposureMode` and `GetTestMode`. `AcquireImage` returns an `SLBufferInfo` whose error is on `.error`. Its `timeout` defaults to 1000 ms, so `CameraDevice.acquire` passes exposure + 2 s; long exposures time out otherwise.
- The exposure getter reports 10 µs units, not the ms that was set.
- Configure first, then stream. Call `SetExposureTime`, `SetNumberOfFrames` and `SetFullWell` before `StartStream()`. Always pair `StartStream()` with `StopStream()` in `try/finally`. Once after `OpenCamera()`, call `SetExposureMode(ExposureModes.seq_mode)` and `SetDDS(False)`.
- `SLImage` is a stack of frames. `Frame2Array(0)` returns only the first frame.
- Data is 14-bit in a uint16 container (max 16383). Invert as `2**14 - arr` and clip corrections to `[0, 2**14 - 1]`.
- Corrections (`OffsetCorrection`, `GainCorrection`, `DefectCorrection`) require matching dimensions, otherwise `SL_ERROR_CORRECTION`. Rotating by 90 or 270 degrees swaps width and height, so don't rotate in place.
- `SetTestMode(True)` gives firmware-generated test images without a sensor.
- `SL_ERROR_ABROTED` is misspelled in the library itself. Use that spelling.
