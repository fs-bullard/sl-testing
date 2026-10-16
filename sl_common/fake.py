"""Hardware-free stand-in for SLDevicePythonWrapper, for developing off Windows.

Implements only the surface CameraDevice uses. Extend as tests need more.
"""
import types
from enum import Enum

import numpy as np


class SLError(Enum):
    SL_ERROR_SUCCESS = 0
    SL_ERROR_NOT_FOUND = 1


class FullWellModes(Enum):
    High = 0
    Low = 1


class FakeSLImage:
    def __init__(self, x, y, n):
        self.frames = np.zeros((n, y, x), dtype=np.uint16)

    def Frame2Array(self, i):
        return self.frames[i]


class FakeSLDevice:
    is_fake = True
    sdk = types.SimpleNamespace(SLImage=FakeSLImage, FullWellModes=FullWellModes, SLError=SLError)

    def __init__(self):
        self._ok = SLError.SL_ERROR_SUCCESS

    def OpenCamera(self, *a): return self._ok
    def CloseCamera(self): return self._ok
    def SetExposureTime(self, *a): return self._ok
    def SetNumberOfFrames(self, *a): return self._ok
    def SetFullWell(self, **kw): return self._ok
    def StartStream(self): return self._ok
    def StopStream(self): return self._ok
    def SoftwareTrigger(self): return self._ok

    def AcquireImage(self, image, frame=0):
        rng = np.random.default_rng(frame)
        image.frames[frame] = rng.integers(0, 2**14, image.frames[frame].shape, dtype=np.uint16)
        return types.SimpleNamespace(error=self._ok)
