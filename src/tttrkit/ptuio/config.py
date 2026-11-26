"""Scan configuration and the data-quality warnings issued against it."""

import numpy as np

FRAME_SYNC_MODES = ("count", "validate", "resync")


class DataQualityWarning(UserWarning):
    """The data and the configuration disagree enough to lose information."""


class FrameSyncWarning(DataQualityWarning):
    """Frame markers disagree with the line count implied by ScanConfig."""


class ScanConfigWarning(DataQualityWarning):
    """The first frame of the data does not match the ScanConfig.

    Issued by :func:`tttrkit.ptuio.prescan.prescan_first_frame` and, for the
    line duration, by SegmentReconstructor.
    """


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
    it from its own chunk when None.

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
        lines: int = 512,
        pixels: int = 512,
        frames: int = 1,
        max_detector: int = 64,
        line_accumulations: tuple = (1,),
        bidirectional: bool = False,
        frame_start_marker_channel: int = 4,
        line_start_marker_channel: int = 1,
        line_stop_marker_channel: int = 2,
        harmonic_scan: bool = False,
        laser_duty: float = 0.6, # used for harmonic correction
        line_start_marker_delay: int = 0, # in nsync units
        line_stop_marker_delay: int = 0,
        line_duration: int | None = None,
        frame_sync: str = "resync",  # "count", "validate" or "resync"
        frame_sync_tolerance: float = 0.1,  # fraction of lines_per_frame
    ):
        if frame_sync not in FRAME_SYNC_MODES:
            raise ValueError(
                f"frame_sync must be one of {FRAME_SYNC_MODES}, got {frame_sync!r}"
            )
        if not 0 < frame_sync_tolerance < 0.5:
            raise ValueError("frame_sync_tolerance must be in (0, 0.5)")
        self.frame_sync = frame_sync
        self.frame_sync_tolerance = frame_sync_tolerance
        self.lines = lines
        self.pixels = pixels
        self.frames = frames
        self.max_detector = max_detector
        self.harmonic_scan = harmonic_scan
        self.laser_duty = laser_duty
        self.line_start_marker_delay = line_start_marker_delay
        self.line_stop_marker_delay = line_stop_marker_delay
        self.line_duration = line_duration

        # Normalize line_accumulations to tuple
        if isinstance(line_accumulations, int):
            self.line_accumulations = (line_accumulations,)
        else:
            self.line_accumulations = tuple(line_accumulations)

        self.num_sequences = len(self.line_accumulations)
        self.bidirectional = bidirectional
        self._total_accumulations = sum(self.line_accumulations)

        self.frame_start_marker_channel = self._validate_marker_mask(
            frame_start_marker_channel, "frame_start_marker_channel"
        )
        self.line_start_marker_channel = self._validate_marker_mask(
            line_start_marker_channel, "line_start_marker_channel"
        )
        self.line_stop_marker_channel = self._validate_marker_mask(
            line_stop_marker_channel, "line_stop_marker_channel"
        )

    @staticmethod
    def _validate_marker_mask(marker_mask: int, name: str) -> int:
        """Validate one physical marker-input bit used for a scan role."""
        if not isinstance(marker_mask, (int, np.integer)):
            raise TypeError(f"{name} must be an integer marker bit")
        if marker_mask not in (1, 2, 4, 8):
            raise ValueError(f"{name} must be one of 1, 2, 4, or 8")
        return int(marker_mask)

    @property
    def lines_per_frame(self) -> int:
        """Number of line start markers per frame (lines x accumulations)."""
        return self.lines * self._total_accumulations

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
            "max_detector": self.max_detector,
            "line_accumulations": self.line_accumulations,
            "bidirectional": self.bidirectional,
            "frame_start_marker_channel": self.frame_start_marker_channel,
            "line_start_marker_channel": self.line_start_marker_channel,
            "line_stop_marker_channel": self.line_stop_marker_channel,
            "harmonic_scan": self.harmonic_scan,
            "laser_duty": self.laser_duty,
            "line_start_marker_delay": self.line_start_marker_delay,
            "line_stop_marker_delay": self.line_stop_marker_delay,
            "line_duration": self.line_duration,
            "frame_sync": self.frame_sync,
            "frame_sync_tolerance": self.frame_sync_tolerance,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            lines=data.get("lines", 512),
            pixels=data.get("pixels", 512),
            frames=data.get("frames", 1),
            max_detector=data.get("max_detector", 64),
            line_accumulations=data.get("line_accumulations", (1,)),
            bidirectional=data.get("bidirectional", False),
            frame_start_marker_channel=data.get("frame_start_marker_channel", 4),
            line_start_marker_channel=data.get("line_start_marker_channel", 1),
            line_stop_marker_channel=data.get("line_stop_marker_channel", 2),
            harmonic_scan=data.get("harmonic_scan", False),
            laser_duty=data.get("laser_duty", 0.6),
            line_start_marker_delay=data.get("line_start_marker_delay", 0),
            line_stop_marker_delay=data.get("line_stop_marker_delay", 0),
            line_duration=data.get("line_duration", None),
            frame_sync=data.get("frame_sync", "resync"),
            frame_sync_tolerance=data.get("frame_sync_tolerance", 0.1),
        )

    def __repr__(self):
        return f"ScanConfig({self.to_dict()})"
