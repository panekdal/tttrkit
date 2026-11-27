"""Scan configuration and the data-quality warnings issued against it.

Defaults are only given where one value fits most setups and a wrong one is
reported (marker channels, frame sync) or where it switches a feature off
(bidirectional, harmonic_scan, delays). Values that describe the particular
acquisition, such as lines and pixels, are required.
"""

import numpy as np

from .decoder import DETECTOR_CHANNELS, check_marker_bit

FRAME_SYNC_MODES = ("count", "validate", "resync")

# Marker inputs of most setups (Abberior, Zeiss, Leica with PicoQuant
# hardware); configure them when the wiring differs. A marker combining a
# configured input with an unconfigured one is discarded and counted, and
# ImageReconstructor warns when that is frequent.
DEFAULT_FRAME_START_MARKER_CHANNEL = 4
DEFAULT_LINE_START_MARKER_CHANNEL = 1
DEFAULT_LINE_STOP_MARKER_CHANNEL = 2


class DataQualityWarning(UserWarning):
    """The data and the configuration disagree enough to lose information."""


class FrameSyncWarning(DataQualityWarning):
    """Frame markers disagree with the line count implied by ScanConfig."""


class ScanConfigWarning(DataQualityWarning):
    """The first frame of the data does not match the ScanConfig.

    Issued by :func:`tttrkit.ptuio.prescan.prescan_first_frame` and, for the
    line duration, by SegmentReconstructor.
    """


def _marker_channel(name: str) -> property:
    """A ScanConfig attribute holding one physical marker-input bit."""
    attr = f"_{name}"

    def fset(self, value: int) -> None:
        setattr(self, attr, check_marker_bit(name, value))

    return property(
        lambda self: getattr(self, attr),
        fset,
        doc="Marker input bit (1, 2, 4 or 8).",
    )


class ScanConfig:
    """Configuration for image reconstruction.

    ``line_start_marker_delay`` and ``line_stop_marker_delay`` are in nsync
    (laser sync) units. A positive start delay begins accepting photons
    later, while a negative value begins earlier. A positive stop delay
    accepts photons farther toward the next line, while a negative value ends
    acceptance earlier. If the two delays differ, the reconstructed line is
    effectively stretched or compressed in phase.

    ``line_duration`` (nsync units) is the photon-acceptance window of a line
    before delays are applied. ImageReconstructor requires it; if None,
    :func:`tttrkit.ptuio.prescan.prescan_first_frame` estimates it from the
    first frame of the data and fills it in. SegmentReconstructor estimates
    it from its own chunk when None. Without line stop markers the estimate
    is a guess (a fraction of the line period) and a ScanConfigWarning says
    so.

    ``frames`` is the number of frames reconstructed; reading stops after
    them, with a DataQualityWarning if more lines follow.

    ``max_detector_channels`` is the number of detector channels kept:
    photons on channel >= max_detector_channels are discarded (counted, and
    warned when frequent). The default keeps every channel the record format
    can hold. It only bounds the output: the reconstruction holds as many
    channels as the data has photons on, which can be fewer.

    ``scan_duty`` (fraction of the line period during which the beam
    sweeps the image) is required with ``harmonic_scan``, and only used
    there.

    Frame assignment:
        Lines are assigned to frames by counting line start markers: line k
        belongs to frame k // lines_per_frame, where
        lines_per_frame = lines * sum(line_accumulations). The configuration
        is the source of truth; frame markers are redundancy used according
        to ``frame_sync``:

        "count": frame markers are ignored.
        "validate": frame markers are checked against the line count and a
            FrameSyncWarning is issued when one is off a frame boundary. The
            reconstruction is not altered.
        "resync" (default): like "validate", but a frame marker that is off
            a frame boundary by at most
            max(1, frame_sync_tolerance * lines_per_frame) lines realigns the
            line count, so that the next line starts a new frame. A dropped
            or spurious line marker then affects only the frame it occurs in.
            Lines that overran the previous frame (spurious line markers) are
            discarded. Markers further off are treated as spurious and
            ignored.

        The check works whether the hardware emits the frame marker before
        the first line of a frame (e.g. Abberior, Zeiss) or after the last
        one (e.g. Leica SP8): either way, the number of line starts preceding
        the marker is a multiple of lines_per_frame.
    """

    def __init__(
        self,
        lines: int,
        pixels: int,
        frames: int = 1,
        max_detector_channels: int = DETECTOR_CHANNELS,
        line_accumulations: tuple = (1,),
        bidirectional: bool = False,
        frame_start_marker_channel: int = DEFAULT_FRAME_START_MARKER_CHANNEL,
        line_start_marker_channel: int = DEFAULT_LINE_START_MARKER_CHANNEL,
        line_stop_marker_channel: int = DEFAULT_LINE_STOP_MARKER_CHANNEL,
        harmonic_scan: bool = False,
        scan_duty: float | None = None,  # required with harmonic_scan
        line_start_marker_delay: int = 0, # in nsync units
        line_stop_marker_delay: int = 0,
        line_duration: int | None = None,
        frame_sync: str = "resync",  # "count", "validate" or "resync"
        frame_sync_tolerance: float = 0.1,  # fraction of lines_per_frame
    ):
        # Validated and normalized attributes are properties; their setters
        # also check later assignments
        self.frame_sync = frame_sync
        self.frame_sync_tolerance = frame_sync_tolerance
        self.lines = lines
        self.pixels = pixels
        self.frames = frames
        self.max_detector_channels = max_detector_channels
        self.harmonic_scan = harmonic_scan
        self.scan_duty = scan_duty
        self.line_start_marker_delay = line_start_marker_delay
        self.line_stop_marker_delay = line_stop_marker_delay
        self.line_duration = line_duration
        self.line_accumulations = line_accumulations
        self.bidirectional = bidirectional
        self.frame_start_marker_channel = frame_start_marker_channel
        self.line_start_marker_channel = line_start_marker_channel
        self.line_stop_marker_channel = line_stop_marker_channel
        self.check_harmonic_scan()

    def check_harmonic_scan(self) -> None:
        """Raise if harmonic_scan is set without a scan_duty."""
        if self.harmonic_scan and self.scan_duty is None:
            raise ValueError(
                "harmonic_scan=True requires scan_duty (the fraction of the "
                "line period the beam sweeps the image); it cannot be "
                "estimated from the data."
            )

    @property
    def scan_duty(self) -> float | None:
        """Sweeping fraction of the line period, for harmonic_scan."""
        return self._laser_duty

    @scan_duty.setter
    def scan_duty(self, value: float | None) -> None:
        if value is not None and not 0 < value <= 1:
            raise ValueError(f"scan_duty must be in (0, 1], got {value}")
        self._laser_duty = value

    frame_start_marker_channel = _marker_channel("frame_start_marker_channel")
    line_start_marker_channel = _marker_channel("line_start_marker_channel")
    line_stop_marker_channel = _marker_channel("line_stop_marker_channel")

    @property
    def frame_sync(self) -> str:
        """How frame markers are used: "count", "validate" or "resync"."""
        return self._frame_sync

    @frame_sync.setter
    def frame_sync(self, value: str) -> None:
        if value not in FRAME_SYNC_MODES:
            raise ValueError(
                f"frame_sync must be one of {FRAME_SYNC_MODES}, got {value!r}"
            )
        self._frame_sync = value

    @property
    def frame_sync_tolerance(self) -> float:
        """Largest frame marker mismatch "resync" corrects, as a fraction of
        lines_per_frame."""
        return self._frame_sync_tolerance

    @frame_sync_tolerance.setter
    def frame_sync_tolerance(self, value: float) -> None:
        if not 0 < value < 0.5:
            raise ValueError("frame_sync_tolerance must be in (0, 0.5)")
        self._frame_sync_tolerance = value

    @property
    def line_accumulations(self) -> tuple[int, ...]:
        """Accumulations of each scan sequence; an int is one sequence."""
        return self._line_accumulations

    @line_accumulations.setter
    def line_accumulations(self, value) -> None:
        if isinstance(value, (int, np.integer)):
            value = (value,)
        value = tuple(int(a) for a in value)
        if not value or min(value) < 1:
            raise ValueError(
                f"line_accumulations must be positive integers, got {value}"
            )
        self._line_accumulations = value

    @property
    def num_sequences(self) -> int:
        return len(self.line_accumulations)

    @property
    def total_accumulations(self) -> int:
        return sum(self.line_accumulations)

    @property
    def lines_per_frame(self) -> int:
        """Number of line start markers per frame (lines x accumulations)."""
        return self.lines * self.total_accumulations

    @property
    def frame_sync_tolerance_lines(self) -> int:
        """Largest frame marker mismatch (in lines) that "resync" corrects:
        max(1, frame_sync_tolerance x lines_per_frame), rounded down."""
        return max(1, int(self.frame_sync_tolerance * self.lines_per_frame))

    def to_dict(self):
        return {
            "lines": self.lines,
            "pixels": self.pixels,
            "frames": self.frames,
            "max_detector_channels": self.max_detector_channels,
            "line_accumulations": self.line_accumulations,
            "bidirectional": self.bidirectional,
            "frame_start_marker_channel": self.frame_start_marker_channel,
            "line_start_marker_channel": self.line_start_marker_channel,
            "line_stop_marker_channel": self.line_stop_marker_channel,
            "harmonic_scan": self.harmonic_scan,
            "scan_duty": self.scan_duty,
            "line_start_marker_delay": self.line_start_marker_delay,
            "line_stop_marker_delay": self.line_stop_marker_delay,
            "line_duration": self.line_duration,
            "frame_sync": self.frame_sync,
            "frame_sync_tolerance": self.frame_sync_tolerance,
        }

    @classmethod
    def from_dict(cls, data):
        """Inverse of to_dict. Missing keys take the defaults of __init__;
        unknown keys raise TypeError."""
        return cls(**data)

    def __repr__(self):
        return f"ScanConfig({self.to_dict()})"
