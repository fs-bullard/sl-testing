"""Non-destructive readout (NDR) backends.

An NDR acquisition is one reset followed by n reads of the same integration, so each read
should hold more signal than the last (an "up-the-ramp" sequence). Both backends return
(reads, read_times_ms): reads is uint16 (n, y, x), read_times_ms is the integration time at
each read, measured from the reset.

NDR is switched on by writing 0 to register 40 and off by writing 1 (NDR_REGISTER).
SdkNdr is the real-hardware path. How the reads arrive once NDR is on is still an assumption:
see TODO(NDR) in SdkNdr.acquire. FakeNdr simulates the ramp on FakeSLDevice so capture and
analysis run without hardware; it makes the same register writes.
"""
import numpy as np

from sl_common.fake import FULL_SCALE

NDR_REGISTER = 40
NDR_ON, NDR_OFF = 0, 1  # note: 0 enables, 1 disables


def nominal_read_times(n_reads, interval_ms):
    """Assumes the first read happens one interval after the reset."""
    return [interval_ms * (k + 1) for k in range(n_reads)]


class _RegisterSwitch:
    """Turns NDR on/off through register 40; shared by the real and fake backends."""

    def __init__(self, cam):
        self.cam = cam
        self.settings = {"register": NDR_REGISTER, "on_value": NDR_ON, "off_value": NDR_OFF}

    def configure(self, n_reads, interval_ms, full_well="High"):
        """Put the detector into NDR mode with n_reads reads, interval_ms apart."""
        self.cam.register_write(NDR_REGISTER, NDR_ON)
        self.settings.update(n_reads=n_reads, interval_ms=interval_ms, full_well=full_well, enabled=True)

    def disable(self):
        """Return to normal destructive readout. Called in finally blocks, so it never raises."""
        try:
            self.cam.register_write(NDR_REGISTER, NDR_OFF)
            self.settings["enabled"] = False
        except Exception as e:
            print(f"WARNING: failed to disable NDR (write {NDR_OFF} to register {NDR_REGISTER}): {e}")


class SdkNdr(_RegisterSwitch):
    """NDR on a real detector through SLDevicePythonWrapper."""

    def acquire(self, n_reads, interval_ms):
        """One reset, n_reads non-destructive reads. Returns (reads uint16 (n, y, x), read_times_ms)."""
        # TODO(NDR): unverified assumption: with NDR on, one triggered sequence of n_reads frames returns
        # successive non-destructive reads of a single integration, one per exposure period. Check it on
        # hardware: analyze's min_increment_z should be clearly positive under light. If the reads arrive
        # differently, change this method only.
        cam, dev, sdk = self.cam, self.cam.device, self.cam._sdk()
        x, y = cam.dims()
        cam._check(dev.SetExposureTime(interval_ms), "set NDR read interval")
        cam._check(dev.SetNumberOfFrames(n_reads), "set number of NDR reads")
        cam._check(dev.SetFullWell(fullWell=getattr(sdk.FullWellModes, self.settings["full_well"])), "set full well")
        cam._check(dev.StartStream(), "start stream")
        try:
            cam._check(dev.SoftwareTrigger(), "software trigger")
            image = sdk.SLImage(x, y, n_reads)
            for i in range(n_reads):
                info = dev.AcquireImage(image, frame=i, timeout=int(interval_ms) + 2000)
                cam._check(info.error, f"NDR read {i}")  # GigE may report MISSING_PACKETS / RESENDS
        finally:
            dev.StopStream()
        reads = np.stack([image.Frame2Array(i).copy() for i in range(n_reads)])
        # TODO(NDR): use hardware timestamps if SLBufferInfo carries them
        return reads, nominal_read_times(n_reads, interval_ms)


class FakeNdr(_RegisterSwitch):
    """Simulated up-the-ramp reads using FakeSLDevice's scene (LED state, dark current, saturating spot)."""

    def __init__(self, cam):
        super().__init__(cam)
        self.settings["simulated"] = True

    def acquire(self, n_reads, interval_ms):
        dev = self.cam.device
        x, y = self.cam.dims()
        times = nominal_read_times(n_reads, interval_ms)
        rng = np.random.default_rng()
        level = dev.signal(0, y, x)  # offset at reset
        reads, prev_t = [], 0
        for t in times:
            mean_step = np.maximum(dev.signal(t, y, x) - dev.signal(prev_t, y, x), 0)
            level = level + rng.normal(mean_step, np.sqrt(mean_step * dev.GAIN_ADU_PER_E))  # charge accumulates
            read = level + rng.normal(0, dev.READ_NOISE_ADU, level.shape)  # read noise is fresh each read
            reads.append(np.clip(read, 0, FULL_SCALE).astype(np.uint16))
            prev_t = t
        return np.stack(reads), times
