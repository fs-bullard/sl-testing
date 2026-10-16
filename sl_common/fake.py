"""Hardware-free stand-in for SLDevicePythonWrapper, for developing off Windows.

Implements only the surface CameraDevice uses. Set `led_on` to simulate the blooming
rig: a bright core that saturates, faint glare, and a vertical bloom streak below it.
"""
import types
from enum import Enum

import numpy as np

FULL_SCALE = 2**14 - 1


class SLError(Enum):
    SL_ERROR_SUCCESS = 0
    SL_ERROR_NOT_FOUND = 1


class FullWellModes(Enum):
    High = 0
    Low = 1


class ExposureModes(Enum):
    seq_mode = 0


class FakeSLImage:
    def __init__(self, x, y, n):
        self.frames = np.zeros((n, y, x), dtype=np.uint16)

    def Frame2Array(self, i):
        return self.frames[i]


class FakeSLDevice:
    is_fake = True
    sdk = types.SimpleNamespace(
        SLImage=FakeSLImage, FullWellModes=FullWellModes, ExposureModes=ExposureModes, SLError=SLError
    )
    X, Y = 256, 384

    def __init__(self):
        self._ok = SLError.SL_ERROR_SUCCESS
        self.led_on = False
        self.exposure_ms = 0
        self.temp = 25.0

    def OpenCamera(self, *a): return self._ok
    def CloseCamera(self): return self._ok
    def SetExposureMode(self, *a): return self._ok
    def SetDDS(self, *a): return self._ok
    def SetNumberOfFrames(self, *a): return self._ok
    def SetFullWell(self, **kw): return self._ok
    def StartStream(self): return self._ok
    def StopStream(self): return self._ok
    def SoftwareTrigger(self): return self._ok
    def GetImageXDim(self): return self.X
    def GetImageYDim(self): return self.Y
    def MeasureTemperature(self, n): return (self._ok, self.temp)

    def SetExposureTime(self, ms):
        self.exposure_ms = ms
        return self._ok

    def AcquireImage(self, image, frame=0):
        t = self.exposure_ms
        h, w = image.frames[frame].shape
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = h * 0.6, w * 0.5
        raw = 100 + 0.002 * t * (1 + 0.05 * (self.temp - 25)) + np.random.normal(0, 3, (h, w))
        if self.led_on:
            flux = 40 * t  # ADU the core would reach without saturating
            r2 = (xx - cx) ** 2 + (yy - cy) ** 2
            glare = 1e-4 * flux * np.exp(-r2 / (2 * 150**2))
            dy = yy - cy
            streak = np.exp(-np.abs(dy) / 60) * np.exp(-((xx - cx) ** 2) / (2 * 8**2)) * np.where(dy > 0, 1, 0.3)
            bloom = 5e-5 * max(flux - FULL_SCALE, 0) * streak
            raw = raw + np.minimum(flux * np.exp(-r2 / (2 * 12**2)), FULL_SCALE) + glare + bloom
            self.temp += 0.002 * t / 1000
        image.frames[frame] = np.clip(raw, 0, FULL_SCALE).astype(np.uint16)
        return types.SimpleNamespace(error=self._ok)
