from collections.abc import Sequence
import copy
import warnings
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from .config import (  # noqa: F401 (re-exported for existing imports)
    DEFAULT_FRAME_START_MARKER_CHANNEL,
    DEFAULT_LINE_START_MARKER_CHANNEL,
    DEFAULT_LINE_STOP_MARKER_CHANNEL,
    DataQualityWarning,
    FrameSyncWarning,
    ScanConfig,
    ScanConfigWarning,
)
from .decoder import (
    DETECTOR_CHANNELS,
    DTIME_CHANNELS,
    MARKER_VALUE_LIMIT,
    check_marker_bit,
    event_dtype,
    photon_events,
    resolve_markers,
)
from .marker_timing import (
    estimate_line_duration,
    frame_boundary_mismatch,
    guessed_line_duration_issue,
    line_duration_issue,
    within_frame_intervals,
)

segment_dtype = [
    ("start_nsync", "i8"),
    ("stop_nsync", "i8"),
    ("frame_idx", "i4"),
    ("line_idx", "i4"),
    ("reversed", "?"),
]

AVAILABLE_OUTPUTS = [
    "photon_count",
    "mean_arrival_time",
    "phasor",
    "tcspc_histogram",
]

# Marker roles in the order resolve_markers returns them; TraceReconstructor
# names its outputs after them
MARKER_ROLES = ("frame_start_marker", "line_start_marker", "line_stop_marker")

# Discarded fractions above which a DataQualityWarning is issued. Below
# them, discarded photons and markers are typically corrupted records
# (impossible dtime or detector values, coincident marker glitches) and are
# only counted, in the attrs of the reconstructed Dataset.
PHOTON_DISCARD_WARN_FRACTION = 1e-3  # of photons inside line windows
MARKER_DISCARD_WARN_FRACTION = 1e-2  # of all marker events


def _build_line_windows(
    line_start_nsyncs: NDArray[np.uint64],
    *,
    line_duration: int,
    line_idx: NDArray[np.int64],
    bidirectional: bool,
    line_start_marker_delay: int, # nsync units
    line_stop_marker_delay: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build photon-acceptance windows for a set of scan lines.

    ``line_start_nsyncs`` contains one entry per complete line, rather than
    the additional terminal marker used to establish line completeness.
    """
    start = line_start_nsyncs.astype(np.int64, copy=True)
    stop = start + line_duration

    start += line_start_marker_delay
    stop += line_stop_marker_delay
    reversed_mask = bidirectional & (line_idx % 2 == 1)

    return start, stop, reversed_mask


def _harmonic_correction(t: np.ndarray, scan_duty: float) -> np.ndarray:
    y = (
        np.cos(0.5 * np.pi * (1 - scan_duty))
        - np.cos(np.pi * scan_duty * t + 0.5 * np.pi * (1 - scan_duty))
    ) / (np.pi * scan_duty)

    I = 2 * np.sin(0.5 * np.pi * scan_duty) / (np.pi * scan_duty)
    return y / I


def _phase_to_pixels(
    nsync: np.ndarray,
    segment_starts: np.ndarray,
    segment_stops: np.ndarray,
    pixels: int,
    reversed_flags: np.ndarray,
    harmonic_scan: bool,
    scan_duty: float | None,
) -> np.ndarray:
    """Convert photon nsync values within their line segment to a pixel index,
    flipping backward (reversed) lines.

    Shared by :class:`ImageReconstructor` and :class:`SegmentReconstructor`.
    """
    phase = (nsync.astype(np.int64) - segment_starts) / (
        segment_stops - segment_starts
    )
    if harmonic_scan:
        phase = _harmonic_correction(phase, scan_duty)
    pixel_idx = np.floor(phase * pixels).astype(int)
    return np.where(reversed_flags, pixels - 1 - pixel_idx, pixel_idx)



class ImageReconstructor:
    def __init__(
        self,
        config: ScanConfig,
        *,
        sync_rate: float,
        tcspc_resolution: float,
        roi_mask: np.ndarray | None = None,
        outputs: Sequence[str] | None = None,
        tcspc_channels: int = DTIME_CHANNELS,
        tcspc_bin_factor: int = 1,
    ):
        """
        Initialize an image reconstructor.

        Parameters:
            config (ScanConfig): Scan configuration object. It is copied:
                changing it afterwards does not affect this reconstructor.

            sync_rate (float): Sync rate in Hz (``reader.header.sync_rate``),
                usually the laser repetition rate. With tcspc_resolution it
                gives the angular frequency of the phasor.

            tcspc_resolution (float): Raw TCSPC resolution in seconds per
                dtime channel (``reader.header.tcspc_resolution``). The unit
                of dtime and of mean_arrival_time; reported as-is in the
                output. The histogram bins are tcspc_bin_factor times wider
                (output tcspc_histogram_resolution).

            roi_mask (ndarray, optional): Binary mask of shape (lines, pixels)
                to restrict processing to specific regions.

            outputs (list of str, optional): List of outputs to compute.
                Each item must be one of the following:
                - "photon_count"
                - "mean_arrival_time"
                - "phasor"
                - "tcspc_histogram"
                If None, all outputs are computed.

            tcspc_channels (int): Number of raw dtime channels kept; photons
                with dtime >= tcspc_channels are discarded (with a
                DataQualityWarning if many). Cannot exceed 2**15, since dtime
                has 15-bit resolution (the default, which keeps every
                photon). The histogram has tcspc_channels / tcspc_bin_factor
                bins.

            tcspc_bin_factor (int): Number of raw TCSPC channels to combine
                into a single histogram bin (e.g. 4 combines every 4 channels
                into 1), reducing resolution but also memory and noise for
                over-resolved acquisitions. Only affects tcspc_histogram;
                mean_arrival_time and phasor keep full raw dtime resolution.
                If it does not evenly divide tcspc_channels, tcspc_channels
                is rounded up to the nearest multiple. Default: 1 (no binning).
        """

        if not isinstance(config, ScanConfig):
            raise TypeError("ImageReconstructor requires a ScanConfig object")
        if config.line_duration is None:
            raise ValueError(
                "ScanConfig.line_duration is not set. Run "
                "tttrkit.ptuio.prescan.prescan_first_frame(reader, config) "
                "first: it estimates the line duration from the first frame "
                "of the data and fills it in. Or set it explicitly."
            )
        if config.line_duration <= 0:
            raise ValueError(
                f"ScanConfig.line_duration must be positive, got "
                f"{config.line_duration}"
            )
        config.check_harmonic_scan()
        if sync_rate <= 0 or tcspc_resolution <= 0:
            raise ValueError(
                "sync_rate and tcspc_resolution must be positive"
            )
        if not isinstance(tcspc_bin_factor, (int, np.integer)) or tcspc_bin_factor < 1:
            raise ValueError("tcspc_bin_factor must be a positive integer")
        if tcspc_channels > DTIME_CHANNELS:
            raise ValueError(
                f"tcspc_channels cannot exceed {DTIME_CHANNELS} "
                "(dtime has 15-bit resolution)"
            )
        # Round up to the nearest multiple of tcspc_bin_factor
        tcspc_channels = int(
            np.ceil(tcspc_channels / tcspc_bin_factor) * tcspc_bin_factor
        )
        # A copy: the output arrays and the frame sync state are built for
        # this config, so later changes to the caller's config must not
        # reach a running reconstruction. Reconstruct again for a new one.
        self.config = copy.deepcopy(config)
        self.shape = (
            config.frames,
            config.lines_per_frame,
            config.pixels,
            config.max_detector_channels,
        )
        # Maps a raw (pre-split) line index to its sequence index
        self._sequence_pattern = np.repeat(
            np.arange(config.num_sequences), config.line_accumulations
        )
        self.active_detectors = set()

        if outputs is None:
            outputs = AVAILABLE_OUTPUTS.copy()

        # Validate requested outputs
        invalid = [o for o in outputs if o not in AVAILABLE_OUTPUTS]
        if invalid:
            raise ValueError(
                f"Invalid output(s): {invalid}. Must be in {AVAILABLE_OUTPUTS}"
            )

        self.requested_outputs = set(outputs)
        self._resolve_dependencies()

        self.tcspc_channels = tcspc_channels
        self.tcspc_bin_factor = tcspc_bin_factor
        self.tcspc_histogram_channels = tcspc_channels // tcspc_bin_factor
        # Raw dtime channel width, and the width of a histogram bin
        self.tcspc_resolution = tcspc_resolution
        self.tcspc_histogram_resolution = tcspc_resolution * tcspc_bin_factor
        # Phase per raw dtime channel: phasors are computed from raw dtimes
        self.omega = 2 * np.pi * sync_rate * self.tcspc_resolution
        self.sync_rate = sync_rate

        # Initialize output arrays
        if "arrival_sum" in self._required:
            self.arrival_sum = np.zeros(self.shape, dtype=np.float32)
        if "photon_count" in self._required:
            self.photon_count = np.zeros(self.shape, dtype=np.uint32)
        if "phasor_sum" in self._required:
            self.phasor_sum = np.zeros(self.shape, dtype=np.complex64)
        if "tcspc_hist" in self._required:
            self.tcspc_hist = np.zeros(
                (
                    self.config.frames,
                    self.config.num_sequences,
                    self.config.max_detector_channels,
                    self.tcspc_histogram_channels,
                ),
                dtype=np.uint64,
            )  # existing shape logic

        # Line assembly. A line is turned into a segment (and its photons
        # binned) only once the frame markers that could move or discard it
        # have been seen, i.e. when `_lookahead` later line starts are known.
        # Until then its start stays in `_open_starts`.
        self._open_starts = np.empty((0,), dtype=np.int64)
        # Global index (0, 1, 2, ...) of _open_starts[0]
        self._line_count = 0
        # Photons that may belong to open lines or to lines still to come
        self._pending_photons = np.empty((0,), dtype=event_dtype)
        self._lookahead = (
            config.frame_sync_tolerance_lines
            if config.frame_sync == "resync"
            else 0
        )

        # Frame sync: line k maps to (k - offset) // lines_per_frame, where
        # offset = _offset_values[i] for the last i with _offset_breaks[i] <= k.
        # Offsets change only on a resync.
        self._offset_breaks = np.zeros(1, dtype=np.int64)
        self._offset_values = np.zeros(1, dtype=np.int64)
        # [lo, hi) global line ranges that overran a frame (found on resync)
        self._excess_lines = []
        self._pending_frame_nsyncs = np.empty((0,), dtype=np.int64)
        # One dict per evaluated frame marker: nsync, lines_before, frame,
        # mismatch (lines off the nearest frame boundary), action
        self.frame_sync_events = []

        # Line duration in nsync units
        self.line_duration = int(config.line_duration)

        self._finished = False  # flag: stop processing when max frames reached

        # Data-quality accounting, reported in the Dataset attrs and, above
        # the thresholds, as DataQualityWarning
        self._photons_in_lines = 0  # photons inside a line window
        self._tcspc_overflow = 0  # ... discarded: dtime >= tcspc_channels
        self._detector_overflow = np.zeros(DETECTOR_CHANNELS, dtype=np.int64)  # by channel
        self._markers_total = 0
        self._markers_unconfigured = {}  # marker value -> count
        self._data_beyond_frames = False

        # ROI for masking
        self._roi_mask_stretched = None
        if roi_mask is not None:
            self._roi_mask_stretched = self._stretch_roi_mask(roi_mask)

    def update(self, events: np.ndarray):
        if self._finished:
            return

        if events.dtype != event_dtype:
            raise TypeError(
                f"Expected events with dtype {event_dtype}, got {events.dtype}"
            )

        self._process(events)

    def _process(self, events: np.ndarray) -> None:
        if len(events) == 0:
            return
        last_nsync = int(events["nsync"][-1])
        self._count_markers(events)
        if len(self._pending_photons) > 0:
            events = np.concatenate([self._pending_photons, events])
            self._pending_photons = np.empty((0,), dtype=events.dtype)

        photons = photon_events(events)
        frame_markers, start_markers, _ = self._resolve_markers(events)

        if self.config.frame_sync != "count":
            self._pending_frame_nsyncs = np.append(
                self._pending_frame_nsyncs,
                frame_markers["nsync"].astype(np.int64),
            )

        line_segments = self._build_line_segments(
            start_markers["nsync"].astype(np.int64)
        )

        # Photons from the first open line on wait for later chunks
        carry = self._carry_mask(photons["nsync"], last_nsync)
        self._pending_photons = photons[carry]
        self._assign_photons_to_segments(photons[~carry], line_segments)

    def _count_markers(self, events: np.ndarray) -> None:
        """Count markers, and those resolve_markers discards as glitches."""
        channels = events["channel"][
            (events["special"] != 0) & (events["channel"] < MARKER_VALUE_LIMIT)
        ]
        self._markers_total += len(channels)
        configured = (
            self.config.frame_start_marker_channel
            | self.config.line_start_marker_channel
            | self.config.line_stop_marker_channel
        )
        unconfigured_bits = np.uint8(0xFF ^ configured)
        values, counts = np.unique(
            channels[(channels & unconfigured_bits) != 0], return_counts=True
        )
        for v, n in zip(values.tolist(), counts.tolist()):
            self._markers_unconfigured[v] = self._markers_unconfigured.get(v, 0) + n

    def _resolve_markers(self, events: np.ndarray):
        return resolve_markers(
            events,
            self.config.frame_start_marker_channel,
            self.config.line_start_marker_channel,
            self.config.line_stop_marker_channel,
        )

    def _carry_mask(self, nsync: np.ndarray, last_nsync: int) -> np.ndarray:
        """Photons that may belong to an open line or to a later line.

        All other photons can only belong to lines already turned into
        segments. Photons between the end of the last open line and the end
        of the chunk belong to no line and are dropped, which bounds the
        carry-over when the stream has no line markers for a while.
        """
        n = nsync.astype(np.int64)
        start_delay = self.config.line_start_marker_delay
        # A later line starts after last_nsync; with a negative start delay
        # it accepts photons from before its marker
        later = n >= last_nsync + min(start_delay, 0)
        if len(self._open_starts) == 0:
            return later
        first = self._open_starts[0] + start_delay
        end = (
            self._open_starts[-1]
            + self.line_duration
            + self.config.line_stop_marker_delay
        )
        return (n >= first) & ((n < end) | later)

    def finalize(self):
        self._finalize_frame_sync()
        self._flush_open_lines()

        data = {}
        active_detectors = sorted(self.active_detectors)
        # A file without binned photons yields empty images, not an error
        channels = max(active_detectors, default=0) + 1

        data["tcspc_resolution"] = ((), self.tcspc_resolution)
        data["tcspc_histogram_resolution"] = ((), self.tcspc_histogram_resolution)
        data["omega"] = ((), self.omega)

        if "tcspc_histogram" in self.requested_outputs:
            self.tcspc_hist = self.tcspc_hist[:, :, :channels, :]
            data["tcspc_histogram"] = (
                ("frame", "sequence", "channel", "tcspc_time"),
                self.tcspc_hist,
            )

        if "photon_count" in self._required:
            photon_count = np.zeros(
                shape=(
                    self.config.frames,
                    self.config.num_sequences,
                    self.config.lines,
                    self.config.pixels,
                    channels,
                ),
                dtype=np.uint32,
            )

            pattern = np.repeat(
                np.arange(self.config.num_sequences),
                self.config.line_accumulations,
            )

            # Total pattern applied to all lines
            sequence_pattern = np.tile(
                pattern, self.config.lines
            )  # shape: (total_lines,)

            lines = self.config.lines
            pixels = self.config.pixels


        if "arrival_sum" in self._required:
            arrival_sum = np.zeros_like(photon_count, dtype=np.float32)

        if "phasor_sum" in self._required:
            phasor_sum = np.zeros_like(photon_count, dtype=np.complex64)

        if "photon_count" in self._required:

            for accu_idx in range(self.config.num_sequences):
                seq_line_idx = np.where(sequence_pattern == accu_idx)[0]
                accum = self.config.line_accumulations[accu_idx]
                for f in range(self.config.frames):
                    summed_PC = self._reshape_and_sum(
                        self.photon_count,
                        f,
                        seq_line_idx,
                        lines,
                        accum,
                        pixels,
                        channels,
                    )
                    photon_count[f, accu_idx, :, :, :] = summed_PC
                    if "arrival_sum" in self._required:
                        summed_AS = self._reshape_and_sum(
                            self.arrival_sum,
                            f,
                            seq_line_idx,
                            lines,
                            accum,
                            pixels,
                            channels,
                        )
                        arrival_sum[f, accu_idx, :, :, :] = summed_AS
                    if "phasor_sum" in self._required:
                        summed_PS = self._reshape_and_sum(
                            self.phasor_sum,
                            f,
                            seq_line_idx,
                            lines,
                            accum,
                            pixels,
                            channels,
                        )
                        phasor_sum[f, accu_idx, :, :, :] = summed_PS

        if "photon_count" in self.requested_outputs:
            data["photon_count"] = (
                ("frame", "sequence", "line", "pixel", "channel"),
                photon_count,
            )

        if "mean_arrival_time" in self.requested_outputs:
            with np.errstate(divide="ignore", invalid="ignore"):
                mean_arrival = np.true_divide(
                    arrival_sum, photon_count, dtype=np.float32
                )
                mean_arrival[photon_count == 0] = 0  # set empty pixels to 0
            data["mean_arrival_time"] = (
                ("frame", "sequence", "line", "pixel", "channel"),
                mean_arrival,
            )

        if "phasor" in self.requested_outputs:

            # Normalize phasor
            with np.errstate(divide="ignore", invalid="ignore"):
                norm_phasor = np.true_divide(
                    phasor_sum, photon_count, dtype=np.complex64
                )
                norm_phasor[photon_count == 0] = np.nan + 1j * np.nan

            g = np.real(norm_phasor)
            s = np.imag(norm_phasor)
            data["phasor_g"] = (
                ("frame", "sequence", "line", "pixel", "channel"),
                g,
            )
            data["phasor_s"] = (
                ("frame", "sequence", "line", "pixel", "channel"),
                s,
            )

        all_coords = {
            "frame": np.arange(self.config.frames),
            "sequence": np.arange(self.config.num_sequences),
            "line": np.arange(self.config.lines),
            "pixel": np.arange(self.config.pixels),
            "channel": np.arange(channels),
            "tcspc_time": np.arange(self.tcspc_histogram_channels)
            * self.tcspc_histogram_resolution,
        }

        # Determine used dimensions
        used_dims = set()
        for data_array in data.values():
            dims = data_array[0]  # First item in tuple is dims
            used_dims.update(dims)

        # Only keep relevant coords
        coords = {k: v for k, v in all_coords.items() if k in used_dims}

        print("Reconstruction finished.")

        return xr.Dataset(data, coords=coords, attrs=self._data_quality())

    def _data_quality(self) -> dict:
        """Warn about significant losses; return counts for Dataset.attrs.

        Attrs (plain integers, so the Dataset stays serialisable):
            photons_in_lines            photons inside a line window
            photons_discarded_tcspc     ... of which dtime >= tcspc_channels
            photons_discarded_detector  ... of which channel >= max_detector_channels
            markers_total               marker events read
            markers_discarded           ... combining configured and
                                        unconfigured inputs (glitches)
            frame_markers_off_boundary  frame markers not on a frame boundary
            data_beyond_frames          1 if complete lines followed the
                                        last configured frame (reading
                                        stopped; also warned)
        """
        in_lines = self._photons_in_lines
        tcspc = self._tcspc_overflow
        detector = int(self._detector_overflow.sum())
        unconfigured = sum(self._markers_unconfigured.values())

        if in_lines and tcspc > PHOTON_DISCARD_WARN_FRACTION * in_lines:
            warnings.warn(
                f"{tcspc} of {in_lines} photons ({tcspc / in_lines:.2%}) have "
                f"a TCSPC channel >= tcspc_channels={self.tcspc_channels} and "
                f"were discarded; tcspc_channels is likely too small.",
                DataQualityWarning,
                stacklevel=3,
            )
        busy = {
            int(ch): int(n)
            for ch, n in enumerate(self._detector_overflow)
            if n > PHOTON_DISCARD_WARN_FRACTION * in_lines
        }
        if in_lines and busy:
            warnings.warn(
                f"Detector channel(s) {busy} (photons) are >= "
                f"max_detector_channels={self.config.max_detector_channels} and were "
                f"discarded; max_detector_channels is likely too small.",
                DataQualityWarning,
                stacklevel=3,
            )
        if unconfigured > MARKER_DISCARD_WARN_FRACTION * self._markers_total:
            warnings.warn(
                f"{unconfigured} of {self._markers_total} markers combine a "
                f"configured input with an unconfigured one "
                f"{dict(sorted(self._markers_unconfigured.items()))} "
                f"(value: count) and were discarded; check the marker "
                f"channels in ScanConfig.",
                DataQualityWarning,
                stacklevel=3,
            )
        if self._data_beyond_frames:
            # Complete lines (line start markers), not photons trailing the
            # last line
            warnings.warn(
                f"Line start markers continue after the "
                f"{self.config.frames} configured frame(s); reading stopped "
                f"there. Increase ScanConfig.frames if the data holds more "
                f"frames.",
                DataQualityWarning,
                stacklevel=3,
            )

        return {
            "photons_in_lines": in_lines,
            "photons_discarded_tcspc": tcspc,
            "photons_discarded_detector": detector,
            "markers_total": self._markers_total,
            "markers_discarded": unconfigured,
            "frame_markers_off_boundary": sum(
                e["action"] != "ok" for e in self.frame_sync_events
            ),
            "data_beyond_frames": int(self._data_beyond_frames),
        }

    def _resolve_dependencies(self):
        required = set()

        if "photon_count" in self.requested_outputs:
            required.add("photon_count")
        if "mean_arrival_time" in self.requested_outputs:
            required.update(["photon_count", "arrival_sum"])
        if "phasor" in self.requested_outputs:
            required.update(["photon_count", "phasor_sum"])
        if "tcspc_histogram" in self.requested_outputs:
            required.add("tcspc_hist")

        self._required = required

    def _stretch_roi_mask(self, base_mask: np.ndarray) -> np.ndarray:
        if base_mask.shape != (self.config.lines, self.config.pixels):
            raise ValueError(
                f"Base ROI mask must have shape ({self.config.lines}, {self.config.pixels})"
            )

        # Expand to (frames, sequences, lines, pixels)
        stretched_mask = np.zeros(
            (
                self.config.frames,
                self.config.lines_per_frame,
                self.config.pixels,
            ),
            dtype=bool,
        )

        for f_idx in range(self.config.frames):
            stretched_mask[f_idx, :, :] = np.repeat(
                base_mask, repeats=self.config.total_accumulations, axis=0
            )

        return stretched_mask
       
    def _build_line_segments(self, start_nsyncs: np.ndarray) -> np.ndarray:
        """Turn line starts into segments, keeping the newest ones open.

        The last line start always stays open (its photons may still be
        arriving); in "resync" mode `_lookahead` more stay open, so that a
        frame marker can still discard lines that overran a frame.
        """
        if self._finished:
            return np.empty(0, dtype=segment_dtype)

        starts = np.concatenate([self._open_starts, start_nsyncs])
        if len(starts) == 0:
            return np.empty(0, dtype=segment_dtype)
        # starts[0] has global line index self._line_count
        first_line = self._line_count

        # All line starts preceding these frame markers are known now, so
        # they can be checked (and resync offsets set) before lines are binned
        self._process_frame_markers(starts, first_line)

        n_build = max(len(starts) - 1 - self._lookahead, 0)
        self._open_starts = starts[n_build:]
        if n_build == 0:
            return np.empty(0, dtype=segment_dtype)
        self._line_count += n_build
        segments = self._make_segments(
            starts[:n_build], first_line + np.arange(n_build, dtype=np.int64)
        )
        if len(segments) and segments["frame_idx"][-1] >= self.config.frames:
            # Frame index never decreases: every later line is out of range too
            self._finished = True
            self._data_beyond_frames = True
        return segments[segments["frame_idx"] < self.config.frames]

    def _make_segments(
        self, start: np.ndarray, global_line: np.ndarray
    ) -> np.ndarray:
        """Build segments from line start nsyncs and their global indices.

        Lines mapping before frame 0 and lines that overran a frame are
        dropped; lines at or beyond config.frames are kept so the caller can
        detect the end of the range.
        """
        frame_idx, line_idx = self._frame_and_line(global_line)
        keep = frame_idx >= 0
        for lo, hi in self._excess_lines:
            keep &= (global_line < lo) | (global_line >= hi)
        frame_idx = frame_idx[keep]
        line_idx = line_idx[keep]

        # If bidirectional: even=forward, odd=reversed
        start, stop, reversed_mask = _build_line_windows(
            start[keep],
            line_duration=self.line_duration,
            line_idx=line_idx,
            bidirectional=self.config.bidirectional,
            line_start_marker_delay=self.config.line_start_marker_delay,
            line_stop_marker_delay=self.config.line_stop_marker_delay,
        )

        result = np.empty(len(start), dtype=segment_dtype)
        result["start_nsync"] = start
        result["stop_nsync"] = stop
        result["frame_idx"] = frame_idx
        result["line_idx"] = line_idx
        result["reversed"] = reversed_mask
        return result

    # ------------------------------------------------------------------
    # Frame sync
    # ------------------------------------------------------------------
    def _line_offset(self, global_line):
        pos = np.searchsorted(self._offset_breaks, global_line, side="right") - 1
        return self._offset_values[pos]

    def _frame_and_line(self, global_line):
        """Map global line indices to (frame, line within frame)."""
        rel = global_line - self._line_offset(global_line)
        return np.divmod(rel, self.config.lines_per_frame)

    def _process_frame_markers(
        self, start_nsyncs: np.ndarray, first_line: int
    ) -> None:
        """Check pending frame markers that precede the last known line start."""
        if len(self._pending_frame_nsyncs) == 0:
            return
        pending = np.sort(self._pending_frame_nsyncs)
        ready = pending[pending <= start_nsyncs[-1]]
        self._pending_frame_nsyncs = pending[pending > start_nsyncs[-1]]
        # A marker coinciding with a line start precedes that line
        lines_before = first_line + np.searchsorted(
            start_nsyncs, ready, side="left"
        )
        for nsync, n in zip(ready, lines_before):
            self._check_frame_marker(int(nsync), int(n))

    def _check_frame_marker(self, nsync: int, lines_before: int) -> None:
        """Compare one frame marker with the frame boundary nearest to it."""
        lines_per_frame = self.config.lines_per_frame
        rel = lines_before - int(self._line_offset(lines_before))
        frame, mismatch = frame_boundary_mismatch(rel, lines_per_frame)
        if frame > self.config.frames:
            return  # beyond the reconstructed range

        if mismatch == 0:
            action = "ok"
        elif self.config.frame_sync == "resync" and (
            abs(mismatch) <= self.config.frame_sync_tolerance_lines
        ):
            # The line right after this marker becomes line 0 of `frame`
            action = "resync"
            self._offset_breaks = np.append(self._offset_breaks, lines_before)
            self._offset_values = np.append(
                self._offset_values, lines_before - frame * lines_per_frame
            )
            if mismatch > 0:
                # Lines past the end of the previous frame belong nowhere.
                # They are still open thanks to the lookahead.
                self._excess_lines.append((lines_before - mismatch, lines_before))
        elif self.config.frame_sync == "resync":
            action = "ignored"
        else:
            action = "reported"

        first_problem = action != "ok" and all(
            e["action"] == "ok" for e in self.frame_sync_events
        )
        self.frame_sync_events.append(
            {
                "nsync": nsync,
                "lines_before": lines_before,
                "frame": frame,
                "mismatch": mismatch,
                "action": action,
            }
        )
        if first_problem:
            warnings.warn(
                f"Frame marker at nsync {nsync} is {mismatch:+d} line(s) off "
                f"the boundary of frame {frame} (expected {lines_per_frame} "
                f"line start markers per frame = lines x accumulations); "
                f"action: {action}. Likely causes: a dropped or spurious line "
                f"marker, a frame marker glitch, or wrong 'lines' / "
                f"'line_accumulations'. All markers are listed in "
                f"ImageReconstructor.frame_sync_events.",
                FrameSyncWarning,
                stacklevel=4,
            )

    def _finalize_frame_sync(self) -> None:
        """Check remaining frame markers and the total line count."""
        lines_seen = self._line_count + len(self._open_starts)
        if not self._finished and lines_seen > 0:
            # Markers after the last line start (e.g. Leica end-of-frame)
            for nsync in np.sort(self._pending_frame_nsyncs):
                self._check_frame_marker(int(nsync), lines_seen)
        # If reading stopped early, later markers cannot be placed
        self._pending_frame_nsyncs = np.empty((0,), dtype=np.int64)

        if self.config.frame_sync != "count":
            if not self.frame_sync_events:
                warnings.warn(
                    "No frame markers found on channel "
                    f"{self.config.frame_start_marker_channel}; frames were "
                    "assigned by line counting only. Set frame_sync='count' "
                    "to silence this.",
                    FrameSyncWarning,
                    stacklevel=3,
                )
            off = [e["action"] for e in self.frame_sync_events if e["action"] != "ok"]
            if len(off) > 1:
                counts = {a: off.count(a) for a in sorted(set(off))}
                warnings.warn(
                    f"{len(off)} of {len(self.frame_sync_events)} frame "
                    f"markers were off a frame boundary {counts}.",
                    FrameSyncWarning,
                    stacklevel=3,
                )

        if not self._finished:
            expected = self.config.frames * self.config.lines_per_frame
            lines_used = lines_seen - int(self._line_offset(lines_seen))
            if lines_used < expected:
                warnings.warn(
                    f"Only {lines_used} of the {expected} line start markers "
                    f"expected for {self.config.frames} frame(s) x "
                    f"{self.config.lines_per_frame} lines were found; the "
                    f"remaining lines are empty.",
                    FrameSyncWarning,
                    stacklevel=3,
                )

    def _assign_photons_to_segments(
        self, photons: np.ndarray, segments: np.ndarray
    ) -> None:
        if len(segments) == 0 or photons.size == 0:
            return

        if photons.dtype != event_dtype:
            raise TypeError(
                f"Expected events with dtype {event_dtype}, got {photons.dtype}"
            )

        segment_starts = segments["start_nsync"]
        segment_ends = segments["stop_nsync"]
        segment_index = (
            np.searchsorted(segment_starts, photons["nsync"], side="right") - 1
        )

        in_line = (segment_index >= 0) & (segment_index < len(segment_starts))
        in_line[in_line] &= (
            photons["nsync"][in_line] < segment_ends[segment_index[in_line]]
        )
        # Photons in a line window that cannot be binned are counted, so
        # that losses are reported once at finalize
        tcspc_overflow = in_line & (photons["dtime"] >= self.tcspc_channels)
        detector_overflow = in_line & (
            photons["channel"] >= self.config.max_detector_channels
        )
        self._photons_in_lines += int(np.count_nonzero(in_line))
        self._tcspc_overflow += int(np.count_nonzero(tcspc_overflow))
        if detector_overflow.any():
            self._detector_overflow += np.bincount(
                photons["channel"][detector_overflow], minlength=DETECTOR_CHANNELS
            )[:DETECTOR_CHANNELS]

        valid = in_line & ~tcspc_overflow & ~detector_overflow

        if np.count_nonzero(valid) == 0:
            return

        segment_index = segment_index[valid]
        photons_in_segments = photons[valid]

        frames = segments["frame_idx"][segment_index]
        lines = segments["line_idx"][segment_index]
        reversed_flags = segments["reversed"][segment_index]

        pixels = _phase_to_pixels(
            photons_in_segments["nsync"],
            segment_starts[segment_index],
            segment_ends[segment_index],
            self.config.pixels,
            reversed_flags,
            harmonic_scan=self.config.harmonic_scan,
            scan_duty=self.config.scan_duty,
        )

        valid_pixels = (pixels >= 0) & (pixels < self.config.pixels)
        if np.count_nonzero(valid_pixels) == 0:
            return

        if self._roi_mask_stretched is not None:
            in_roi = self._roi_mask_stretched[frames, lines, pixels]
            valid_pixels = valid_pixels & in_roi

        pixels = pixels[valid_pixels]
        frames = frames[valid_pixels]
        lines = lines[valid_pixels]
        channels = photons_in_segments["channel"][valid_pixels]
        dtimes = photons_in_segments["dtime"][valid_pixels]

        if "arrival_sum" in self._required:
            np.add.at(
                self.arrival_sum, (frames, lines, pixels, channels), dtimes
            )

        if "photon_count" in self._required:
            np.add.at(self.photon_count, (frames, lines, pixels, channels), 1)

        if "phasor_sum" in self._required:
            phasors = np.exp(1j * self.omega * dtimes)
            np.add.at(
                self.phasor_sum, (frames, lines, pixels, channels), phasors
            )

        if "tcspc_hist" in self._required:
            # Only the histogram is binned; arrival time/phasor keep raw resolution
            hist_dtimes = (
                dtimes // self.tcspc_bin_factor
                if self.tcspc_bin_factor > 1
                else dtimes
            )
            sequence = self._sequence_pattern[
                lines % len(self._sequence_pattern)
            ]
            np.add.at(
                self.tcspc_hist,
                (frames, sequence, channels, hist_dtimes),
                1,
            )

        self.active_detectors.update(np.unique(channels))

    def _reshape_and_sum(
        self, array, f_idx, line_indices, lines, accum, pixels, channels
    ):
        sliced = array[f_idx, line_indices, :, :channels]
        reshaped = sliced.reshape(lines, accum, pixels, channels)
        return reshaped.sum(axis=1)

    def _flush_open_lines(self):
        """Bin the lines still open at the end of the stream."""
        if len(self._open_starts) and not self._finished:
            n = len(self._open_starts)
            segments = self._make_segments(
                self._open_starts,
                self._line_count + np.arange(n, dtype=np.int64),
            )
            segments = segments[segments["frame_idx"] < self.config.frames]
            self._assign_photons_to_segments(self._pending_photons, segments)
        self._line_count += len(self._open_starts)
        self._open_starts = np.empty((0,), dtype=np.int64)
        self._pending_photons = np.empty(
            (0,), dtype=self._pending_photons.dtype
        )

class SegmentReconstructor:
    """Reconstruct a single chunk of events into a (line, pixel) image, using
    only the complete lines bounded by line-start markers found within that
    chunk.

    Unlike :class:`ImageReconstructor`, this is a lightweight, single-shot,
    stateless reconstructor: it does not track frames, does not resolve
    channels (all detectors are summed together), and only ever uses the
    first scan sequence. It is intended for quickly probing a small window of
    data, e.g. for bidirectional phase-shift estimation, where reconstructing
    a full multi-frame, per-channel image would be unnecessary overhead.

    The output has as many lines as there are complete line-start-to-line-start
    intervals in the chunk (i.e. ``len(line_start_markers) - 1``), not the
    nominal number of lines in a full frame.

    Since frames aren't tracked, a frame-start marker within the chunk means
    the lines before and after it belong to different frames (a discontinuity).
    Such breaks are reported via the ``frame_break_line`` output variable: line
    index after which a frame-start marker was detected.

    Pixels are always mapped linearly in time, ignoring ``harmonic_scan``:
    the forward/backward alignment it serves needs no harmonic correction,
    and a wrong ``scan_duty`` (which cannot be measured) would bias it.

    The config is copied. A line duration it lacks is estimated from each
    chunk and returned in the ``line_duration`` variable.
    """

    def __init__(self, config: ScanConfig):
        if not isinstance(config, ScanConfig):
            raise TypeError("SegmentReconstructor requires a ScanConfig object")
        self.config = copy.deepcopy(config)

    def reconstruct(self, events: np.ndarray) -> xr.Dataset:
        """Process one chunk of events and return the reconstructed image.

        Args:
            events: Array of events with dtype ``event_dtype``.

        Returns:
            xr.Dataset with a ``photon_count`` variable of dims
            ``("line", "pixel")``, a ``frame_break_line`` variable listing
            the line index after which each frame-start marker found in the
            chunk occurred (empty if the chunk doesn't span a frame boundary)
            and the scalar ``line_duration`` used (nsync units, -1 if there
            was none).
        """
        if events.dtype != event_dtype:
            raise TypeError(
                f"Expected events with dtype {event_dtype}, got {events.dtype}"
            )

        photons = photon_events(events)
        frame_markers, start_markers, stop_markers = resolve_markers(
            events,
            self.config.frame_start_marker_channel,
            self.config.line_start_marker_channel,
            self.config.line_stop_marker_channel,
        )

        if len(start_markers) < 2:
            # Not enough line-start markers to bound a single complete line
            return self._empty_dataset()

        line_duration = self.config.line_duration
        if line_duration is None:
            # A single chunk, so the estimate uses this chunk only. Its line
            # offset within a frame is unknown: only frame markers exclude
            # the intervals between frames. Callers with a reader can run
            # prescan_first_frame instead, as for ImageReconstructor.
            estimate = estimate_line_duration(
                frame_markers["nsync"],
                start_markers["nsync"],
                stop_markers["nsync"],
            )
            if estimate.duration is None:
                return self._empty_dataset()
            issue = guessed_line_duration_issue(estimate)
            if issue:
                warnings.warn(issue, ScanConfigWarning, stacklevel=2)
            line_duration = estimate.duration
        else:
            self._check_line_duration(
                start_markers["nsync"], frame_markers["nsync"]
            )

        if line_duration <= 0:
            return self._empty_dataset()

        start, stop, line_idx, reversed_mask = self._build_segments(
            start_markers["nsync"], line_duration
        )
        photon_count = np.zeros(
            (len(start), self.config.pixels), dtype=np.uint32
        )
        self._assign_photons(photons, start, stop, line_idx, reversed_mask, photon_count)
        frame_break_lines = self._find_frame_breaks(
            frame_markers["nsync"], start_markers["nsync"]
        )

        return xr.Dataset(
            {
                "photon_count": (("line", "pixel"), photon_count),
                "frame_break_line": (("frame_break",), frame_break_lines),
                "line_duration": ((), int(line_duration)),
            },
            coords={
                "line": np.arange(len(start)),
                "pixel": np.arange(self.config.pixels),
                "frame_break": np.arange(len(frame_break_lines)),
            },
        )

    def _check_line_duration(
        self, start_nsyncs: NDArray[np.uint64], frame_nsyncs: NDArray[np.uint64]
    ) -> None:
        """Warn if the given line duration exceeds the chunk's line period."""
        periods = within_frame_intervals(start_nsyncs, frame_nsyncs)
        period = float(np.median(periods)) if len(periods) else None
        issue = line_duration_issue(self.config.line_duration, period)
        if issue:
            warnings.warn(issue, ScanConfigWarning, stacklevel=3)

    def _find_frame_breaks(
        self, frame_nsyncs: NDArray[np.uint64], start_nsyncs: NDArray[np.uint64]
    ) -> np.ndarray:
        """For each frame-start marker, find the index of the line right
        before it (i.e. the last line of the previous frame)."""
        if len(frame_nsyncs) == 0:
            return np.empty(0, dtype=np.int64)

        used_starts = start_nsyncs[:-1].astype(np.int64)
        idx = np.searchsorted(used_starts, frame_nsyncs.astype(np.int64), side="right") - 1
        return np.clip(idx, 0, len(used_starts) - 1)

    def _empty_dataset(self) -> xr.Dataset:
        return xr.Dataset(
            {
                "photon_count": (
                    ("line", "pixel"),
                    np.zeros((0, self.config.pixels), dtype=np.uint32),
                ),
                "frame_break_line": (("frame_break",), np.empty(0, dtype=np.int64)),
                "line_duration": ((), int(self.config.line_duration or -1)),
            },
            coords={
                "line": np.arange(0),
                "pixel": np.arange(self.config.pixels),
                "frame_break": np.arange(0),
            },
        )


    def _build_segments(
        self, start_nsyncs: NDArray[np.uint64], line_duration: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        line_idx = np.arange(len(start_nsyncs) - 1, dtype=np.int64)
        start, stop, reversed_mask = _build_line_windows(
            start_nsyncs[:-1],
            line_duration=line_duration,
            line_idx=line_idx,
            bidirectional=self.config.bidirectional,
            line_start_marker_delay=self.config.line_start_marker_delay,
            line_stop_marker_delay=self.config.line_stop_marker_delay,
        )

        return start, stop, line_idx, reversed_mask

    def _assign_photons(
        self,
        photons: np.ndarray,
        start: np.ndarray,
        stop: np.ndarray,
        line_idx: np.ndarray,
        reversed_mask: np.ndarray,
        photon_count: np.ndarray,
    ) -> None:
        if len(start) == 0 or photons.size == 0:
            return

        segment_index = np.searchsorted(start, photons["nsync"], side="right") - 1

        valid = (
            (segment_index >= 0)
            & (segment_index < len(start))
            & (photons["nsync"] < stop[segment_index])
        )
        if np.count_nonzero(valid) == 0:
            return

        segment_index = segment_index[valid]
        photons_in_segments = photons[valid]

        lines = line_idx[segment_index]
        reversed_flags = reversed_mask[segment_index]

        pixels = _phase_to_pixels(
            photons_in_segments["nsync"],
            start[segment_index],
            stop[segment_index],
            self.config.pixels,
            reversed_flags,
            # Linear on purpose, see the class docstring
            harmonic_scan=False,
            scan_duty=None,
        )

        valid_pixels = (pixels >= 0) & (pixels < self.config.pixels)
        if np.count_nonzero(valid_pixels) == 0:
            return

        np.add.at(photon_count, (lines[valid_pixels], pixels[valid_pixels]), 1)


class TraceReconstructor:
    """Reconstruct time-resolved intensity traces from TTTR data.
    
    This class bins photon events into time bins and optionally extracts
    marker times (frame start, line start, line stop) within the specified
    time window.
    """
    def __init__(
        self,
        *,
        sync_rate: float,
        stop_time_s: float,
        start_time_s: float = 0.0,
        bin_width_s: float = 1e-3,
        max_detector_channels: int = DETECTOR_CHANNELS,
        outputs: Sequence[str] | None = None,
        frame_start_marker_channel: int = DEFAULT_FRAME_START_MARKER_CHANNEL,
        line_start_marker_channel: int = DEFAULT_LINE_START_MARKER_CHANNEL,
        line_stop_marker_channel: int = DEFAULT_LINE_STOP_MARKER_CHANNEL,
    ):
        """
        Initialize a trace reconstructor.

        Parameters:
            sync_rate (float): Sync rate in Hz (``reader.header.sync_rate``).
            stop_time_s (float): End of the trace window in seconds.
            start_time_s (float): Start of the trace window in seconds.
                Default: 0.0
            bin_width_s (float): Time bin width in seconds. Default: 1e-3 (1 ms)
            max_detector_channels (int): Detector channels kept. Default: all the
                record format can hold.
            outputs (list of str, optional): List of outputs to compute.
                - "photon_count": intensity trace per detector
                - "markers": marker times (frame start, line start, line stop)
                If None, both outputs are computed.
            frame_start_marker_channel, line_start_marker_channel,
            line_stop_marker_channel (int): Marker input bits, as in
                ScanConfig.
        """

        if sync_rate <= 0:
            raise ValueError("sync_rate must be positive")
        if bin_width_s <= 0:
            raise ValueError("bin_width_s must be positive")
        if start_time_s >= stop_time_s:
            raise ValueError("start_time_s must be less than stop_time_s")

        self.start_time_s = start_time_s
        self.stop_time_s = stop_time_s
        self.max_detector_channels = max_detector_channels
        self.bin_width_s = bin_width_s
        self.sync_rate = sync_rate
        self.marker_channels = (
            check_marker_bit("frame_start_marker_channel", frame_start_marker_channel),
            check_marker_bit("line_start_marker_channel", line_start_marker_channel),
            check_marker_bit("line_stop_marker_channel", line_stop_marker_channel),
        )

        # Calculate number of bins
        self.n_bins = int(np.ceil((stop_time_s - start_time_s) / bin_width_s))

        # Convert time boundaries to nsync
        self.start_nsync = int(start_time_s * sync_rate)
        self.stop_nsync = int(stop_time_s * sync_rate)
        self.bin_width_nsync = int(bin_width_s * sync_rate)

        # Create time edges and bin centers
        self.time_edges_nsync = (
            np.arange(self.n_bins + 1, dtype=np.float64) * self.bin_width_nsync
        ) + self.start_nsync
        self.time_axis_s = (
            (np.arange(self.n_bins, dtype=np.float64) + 0.5) * self.bin_width_s
            + self.start_time_s
        )

        # Set outputs
        if outputs is None:
            outputs = ["photon_count", "markers"]

        valid_outputs = ["photon_count", "markers"]
        invalid = [o for o in outputs if o not in valid_outputs]
        if invalid:
            raise ValueError(
                f"Invalid output(s): {invalid}. Must be in {valid_outputs}"
            )

        self.requested_outputs = set(outputs)

        # Initialize arrays
        if "photon_count" in self.requested_outputs:
            self.intensity = np.zeros(
                (self.n_bins, max_detector_channels), dtype=np.uint64
            )

        if "markers" in self.requested_outputs:
            # nsync of the markers in the window, per role
            self.marker_nsyncs = {
                role: np.array([], dtype=np.uint64) for role in MARKER_ROLES
            }

        self.active_detectors = set()

    def update(self, events: np.ndarray):
        """Process a chunk of events.
        
        Parameters:
            events (ndarray): Array of events with dtype event_dtype
        """

        if events.dtype != event_dtype:
            raise TypeError(
                f"Expected events with dtype {event_dtype}, got {events.dtype}"
            )

        # Extract and bin photons by time
        if "photon_count" in self.requested_outputs:
            photons = photon_events(events)

            # Find bin indices for each photon
            bin_indices = (
                np.searchsorted(
                    self.time_edges_nsync, photons["nsync"], side="right"
                )
                - 1
            )

            # Validate bins and channels
            valid = (
                (bin_indices >= 0)
                & (bin_indices < self.n_bins)
                & (photons["channel"] < self.max_detector_channels)
            )

            if np.count_nonzero(valid) > 0:
                bin_indices_valid = bin_indices[valid]
                channels_valid = photons["channel"][valid]
                np.add.at(
                    self.intensity,
                    (bin_indices_valid, channels_valid),
                    1,
                )
                self.active_detectors.update(np.unique(channels_valid))

        # Extract markers within the time window
        if "markers" in self.requested_outputs:
            markers = resolve_markers(events, *self.marker_channels)
            for role, found in zip(MARKER_ROLES, markers):
                nsync = found["nsync"]
                in_window = (nsync >= self.start_nsync) & (nsync < self.stop_nsync)
                self.marker_nsyncs[role] = np.append(
                    self.marker_nsyncs[role], nsync[in_window]
                )

    def finalize(self) -> xr.Dataset:
        """Finalize and return results as xarray Dataset.

        Returns:
            xr.Dataset: Dataset containing:
                - photon_count: intensity trace with dimensions (time, channel)
                - frame_start_marker_times_s, line_start_marker_times_s,
                  line_stop_marker_times_s: marker times in seconds, each
                  along its own dimension (frame_start_marker, ...); present,
                  possibly empty, when "markers" was requested
        """

        data = {}
        channels = max(self.active_detectors) + 1 if self.active_detectors else 1
        coords = {
            "time": self.time_axis_s,
            "channel": np.arange(channels),
        }

        if "photon_count" in self.requested_outputs:
            # Trim intensity array to active channels
            intensity = self.intensity[:, :channels]
            data["photon_count"] = (("time", "channel"), intensity)

        if "markers" in self.requested_outputs:
            for role in MARKER_ROLES:
                nsyncs = self.marker_nsyncs[role]
                data[f"{role}_times_s"] = ((role,), nsyncs / self.sync_rate)
                coords[role] = np.arange(len(nsyncs))

        return xr.Dataset(data, coords=coords)



