class CameraError(RuntimeError):
    """An SLDevice call returned a non-success SLError."""


class CameraNotConnectedError(CameraError):
    """SL_ERROR_NOT_FOUND: the detector is absent."""
