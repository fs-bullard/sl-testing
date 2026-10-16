"""Error-checked wrapper around SLDevice. Raises instead of returning SLError codes."""
from .errors import CameraError, CameraNotConnectedError
from .sdk import require_sdk


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

    def open_camera(self, buffer_depth=10):
        self._check(self.device.OpenCamera(buffer_depth), "open camera")

    def close_camera(self):
        self._check(self.device.CloseCamera(), "close camera")

    def acquire(self, x_dim, y_dim, n_frames, exposure_ms, full_well=None):
        """Configure, stream and capture n_frames; returns the SLImage stack."""
        sdk = require_sdk() if not hasattr(self.device, "is_fake") else self.device.sdk
        self._check(self.device.SetExposureTime(exposure_ms), "set exposure time")
        self._check(self.device.SetNumberOfFrames(n_frames), "set number of frames")
        self._check(
            self.device.SetFullWell(fullWell=full_well or sdk.FullWellModes.High),
            "set full well",
        )
        self._check(self.device.StartStream(), "start stream")
        try:
            self._check(self.device.SoftwareTrigger(), "software trigger")
            image = sdk.SLImage(x_dim, y_dim, n_frames)
            for i in range(n_frames):
                info = self.device.AcquireImage(image, frame=i)
                self._check(info.error, f"acquire frame {i}")
            return image
        finally:
            self.device.StopStream()
