"""Non-destructive readout (NDR) backends.

An NDR acquisition is one reset followed by n reads of the same integration, so each read
should hold more signal than the last (an "up-the-ramp" sequence). Both backends return
(reads, read_times_ms): reads is uint16 (n, y, x), read_times_ms is the integration time at
each read, measured from the reset.

SdkNdr is the real-hardware path and is mostly placeholders: search for TODO(NDR).
FakeNdr simulates the ramp on FakeSLDevice so capture and analysis run without hardware.
"""
import numpy as np

from sl_common.fake import FULL_SCALE

# TODO(NDR): register addresses/values (or SDK calls) that switch NDR on and off, once known.
# Whatever configure() writes is logged in meta.json via `settings`.
NDR_REGISTERS_ON = {}
NDR_REGISTERS_OFF = {}


def nominal_read_times(n_reads, interval_ms):
    """Assumes the first read happens one interval after the reset."""
    return [interval_ms * (k + 1) for k in range(n_reads)]


class SdkNdr:
    """NDR on a real detector through SLDevicePythonWrapper. Placeholders until the mechanism is known."""

    def __init__(self, cam):
        self.cam = cam
        self.settings = {}

    def configure(self, n_reads, interval_ms):
        """Put the detector into NDR mode with n_reads reads, interval_ms apart."""
        # TODO(NDR): switch the sensor to non-destructive readout. Candidates:
        #   - register writes:  for addr, value in NDR_REGISTERS_ON.items(): self.cam.register_write(addr, value)
        #   - an SDK call or exposure mode, if the wrapper exposes one (check dir(SLDevice), help(ExposureModes))
        #   - a firmware or .pvcfg option on the Pleora GigE side
        # Also set how many reads per reset and their spacing (interval_ms), if the hardware allows it.
        # Record everything written in self.settings so it lands in meta.json.
        raise NotImplementedError("TODO(NDR): configure NDR mode in tests/ndr/ndr.py")

    def acquire(self, n_reads, interval_ms):
        """One reset, n_reads non-destructive reads. Returns (reads uint16 (n, y, x), read_times_ms)."""
        # TODO(NDR): capture the reads of one integration. One possible shape, if NDR reads arrive as
        # successive frames of a single triggered sequence:
        #
        #   dev, sdk = self.cam.device, self.cam._sdk()
        #   x, y = self.cam.dims()
        #   self.cam._check(dev.SetNumberOfFrames(n_reads), "set number of reads")
        #   self.cam._check(dev.StartStream(), "start stream")
        #   try:
        #       self.cam._check(dev.SoftwareTrigger(), "software trigger")
        #       image = sdk.SLImage(x, y, n_reads)
        #       for i in range(n_reads):
        #           info = dev.AcquireImage(image, frame=i, timeout=int(interval_ms) + 2000)
        #           self.cam._check(info.error, f"NDR read {i}")   # GigE may report MISSING_PACKETS / RESENDS
        #   finally:
        #       dev.StopStream()
        #   reads = np.stack([image.Frame2Array(i).copy() for i in range(n_reads)])
        #
        # Read times: use hardware timestamps if SLBufferInfo carries them, otherwise nominal_read_times().
        raise NotImplementedError("TODO(NDR): acquire NDR reads in tests/ndr/ndr.py")

    def disable(self):
        """Return to normal destructive readout. Called in a finally block, so it must not raise."""
        # TODO(NDR): undo configure(), e.g. write NDR_REGISTERS_OFF. Wrap in try/except and log on failure.
        pass


class FakeNdr:
    """Simulated up-the-ramp reads using FakeSLDevice's scene (LED state, dark current, saturating spot)."""

    READ_NOISE_ADU = 3.0
    GAIN_ADU_PER_E = 0.25  # sets shot noise

    def __init__(self, cam):
        self.cam = cam
        self.settings = {"simulated": True}

    def configure(self, n_reads, interval_ms):
        self.settings.update(n_reads=n_reads, interval_ms=interval_ms)

    def acquire(self, n_reads, interval_ms):
        dev = self.cam.device
        x, y = self.cam.dims()
        times = nominal_read_times(n_reads, interval_ms)
        rng = np.random.default_rng()
        level = dev.signal(0, y, x)  # offset at reset
        reads, prev_t = [], 0
        for t in times:
            mean_step = np.maximum(dev.signal(t, y, x) - dev.signal(prev_t, y, x), 0)
            level = level + rng.normal(mean_step, np.sqrt(mean_step * self.GAIN_ADU_PER_E))  # charge accumulates
            read = level + rng.normal(0, self.READ_NOISE_ADU, level.shape)  # read noise is fresh each read
            reads.append(np.clip(read, 0, FULL_SCALE).astype(np.uint16))
            prev_t = t
        return np.stack(reads), times

    def disable(self):
        pass
