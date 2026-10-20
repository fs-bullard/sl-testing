"""Error-checked wrapper around SLDevice. Raises instead of returning SLError codes."""
from .errors import CameraError, CameraNotConnectedError
from .sdk import require_sdk

ACQUIRE_MARGIN_MS = 2000  # readout and transfer allowance on top of the exposure time


def _value(result):
    """Getters may return (SLError, value) or a bare value."""
    return result[1] if isinstance(result, tuple) else result


def resolve_interface(name):
    """Map 'usb' / 'gige' to a DeviceInterface member (GigE members are named Pleora/GigE)."""
    sdk = require_sdk()
    members = {n: getattr(sdk.DeviceInterface, n) for n in dir(sdk.DeviceInterface) if not n.startswith("_")}
    wanted = ("usb",) if name == "usb" else ("gige", "pleora")
    for n, member in members.items():
        if any(w in n.lower() for w in wanted) and hasattr(member, "name"):
            return member
    raise CameraError(f"No DeviceInterface member matches '{name}'; found {sorted(members)}")


class CameraDevice:
    def __init__(self, device=None, iface=None):
        """Pass `device` to inject a fake; otherwise a real SLDevice is built (USB by default)."""
        if device is None:
            sdk = require_sdk()
            device = sdk.SLDevice(iface if iface is not None else sdk.DeviceInterface.USB)
        self.device = device

    def _check(self, err, op):
        name = getattr(err, "name", str(err))
        if name == "SL_ERROR_SUCCESS":
            return
        if name == "SL_ERROR_NOT_FOUND":
            raise CameraNotConnectedError(f"Failed to {op}: not connected")
        raise CameraError(f"Failed to {op}: {err}")

    def _sdk(self):
        return self.device.sdk if hasattr(self.device, "is_fake") else require_sdk()

    def open_camera(self, buffer_depth=10, pvcfg_path=None):
        if pvcfg_path:  # Pleora (GigE) only; must precede OpenCamera
            self._check(self.device.SetPvcfgPath(pvcfg_path), "set pvcfg path")
        self._check(self.device.OpenCamera(buffer_depth), "open camera")

    def close_camera(self):
        self._check(self.device.CloseCamera(), "close camera")

    def setup(self, dds=False):
        """One-time config after OpenCamera."""
        self._check(self.device.SetExposureMode(self._sdk().ExposureModes.seq_mode), "set exposure mode")
        self._check(self.device.SetDDS(dds), "set DDS")

    def register_write(self, addr, value):
        self._check(self.device.RegisterWrite(addr, value), f"write register {addr}")

    def dims(self):
        """(x_dim, y_dim)"""
        return int(_value(self.device.GetImageXDim())), int(_value(self.device.GetImageYDim()))

    def temperature(self, sensor=0):
        """Sensor temperature, or None if unavailable (logging only, so never fatal)."""
        try:
            return float(_value(self.device.MeasureTemperature(sensor)))
        except Exception:
            return None

    def acquire(self, x_dim, y_dim, n_frames, exposure_ms, full_well="High"):
        """Configure, stream and capture n_frames on one trigger; returns the SLImage stack."""
        sdk = self._sdk()
        self._check(self.device.SetExposureTime(exposure_ms), "set exposure time")
        self._check(self.device.SetNumberOfFrames(n_frames), "set number of frames")
        self._check(
            self.device.SetFullWell(fullWell=getattr(sdk.FullWellModes, full_well)),
            "set full well",
        )
        self._check(self.device.StartStream(), "start stream")
        try:
            self._check(self.device.SoftwareTrigger(), "software trigger")
            image = sdk.SLImage(x_dim, y_dim, n_frames)
            # AcquireImage's timeout defaults to 1000 ms, so long exposures need a longer wait per frame
            timeout_ms = int(exposure_ms) + ACQUIRE_MARGIN_MS
            for i in range(n_frames):
                info = self.device.AcquireImage(image, frame=i, timeout=timeout_ms)
                self._check(info.error, f"acquire frame {i}")
            return image
        finally:
            self.device.StopStream()
