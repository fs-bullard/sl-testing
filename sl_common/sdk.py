"""Import the vendor module from the Python path, or report that it is unavailable.

SLDevicePythonWrapper is Windows-only, so on other machines `sdk` is None and
only the fake device can be used.
"""
try:
    import SLDevicePythonWrapper as sdk
except ImportError:
    sdk = None


def require_sdk():
    if sdk is None:
        raise ImportError(
            "SLDevicePythonWrapper not importable; put the .pyd and its DLLs on the "
            "Python path (Windows only) or run with the fake device."
        )
    return sdk
