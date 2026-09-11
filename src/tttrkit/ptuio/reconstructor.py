from collections.abc import Sequence
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from .decoder import event_dtype, get_markers, get_photons, resolve_markers

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


def _adjust_line_bounds(
    start: np.ndarray,
    stop: np.ndarray,
    reversed_flags: np.ndarray,
    line_duration: int,
    # bidirectional: bool,
    # bidirectional_phase_shift: float,
    line_start_marker_delay: float,
    line_stop_marker_delay: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Shift line start/stop bounds by the configured delays and, for
    bidirectional scans, the phase shift applied to backward lines.

    Shared by :class:`ImageReconstructor` and :class:`SegmentReconstructor` so
    the phase-shift math only needs fixing in one place.
    """
    start = np.asarray(start, dtype=np.int64).copy()
    stop = np.asarray(stop, dtype=np.int64).copy()
    reversed_flags = np.asarray(reversed_flags, dtype=bool)

    line_start_delay = int(line_start_marker_delay * line_duration)
    line_stop_delay = int(line_stop_marker_delay * line_duration)

    start += line_start_delay
    stop += line_stop_delay

    # if bidirectional:
    #     shift = int(bidirectional_phase_shift * line_duration)
    #     start[reversed_flags] += shift
    #     stop[reversed_flags] += shift

    return start, stop


def _harmonic_correction(t: np.ndarray, laser_duty: float) -> np.ndarray:
    y = (
        np.cos(0.5 * np.pi * (1 - laser_duty))
        - np.cos(np.pi * laser_duty * t + 0.5 * np.pi * (1 - laser_duty))
    ) / (np.pi * laser_duty)

    I = 2 * np.sin(0.5 * np.pi * laser_duty) / (np.pi * laser_duty)
    return y / I


def _phase_to_pixels(
    nsync: np.ndarray,
    segment_starts: np.ndarray,
    segment_stops: np.ndarray,
    pixels: int,
    reversed_flags: np.ndarray,
    harmonic_scan: bool = False,
    laser_duty: float = 0.6,
) -> np.ndarray:
    """Convert photon nsync values within their line segment to a pixel index,
    flipping backward (reversed) lines.

    Shared by :class:`ImageReconstructor` and :class:`SegmentReconstructor`.
    """
    phase = (nsync.astype(np.int64) - segment_starts) / (
        segment_stops - segment_starts
    )
    if harmonic_scan:
        phase = _harmonic_correction(phase, laser_duty)
    pixel_idx = np.floor(phase * pixels).astype(int)
    return np.where(reversed_flags, pixels - 1 - pixel_idx, pixel_idx)



class ScanConfig:
    """Configuration for image reconstruction.

    ``line_start_marker_delay`` and ``line_stop_marker_delay`` are
    dimensionless fractions of the inferred line duration. A positive start
    delay begins accepting photons later, while a negative value begins
    earlier. A positive stop delay accepts photons farther toward the next
    line, while a negative value ends acceptance earlier. If the two delays
    differ, the reconstructed line is effectively stretched or compressed in
    phase.
    """

    def __init__(
        self,
        lines: int = 512,
        pixels: int = 512,
        frames: int = 1,
        max_detector: int = 64,
        line_accumulations: tuple = (
            1,
        ),  #  > 1 dimension means the scanning is sequential
        bidirectional: bool = False,
        bidirectional_phase_shift: float = 0.0,
        frame_start_marker_channel: int = 4,
        line_start_marker_channel: int = 1,
        line_stop_marker_channel: int = 2,
        harmonic_scan: bool = False,
        laser_duty: float = 0.6, # used for harmonic correction
        line_start_marker_delay: float = 0,
        line_stop_marker_delay: float = 0,
    ):
        self.lines = lines
        self.pixels = pixels
        self.frames = frames
        self.max_detector = max_detector
        self.bidirectional_phase_shift = bidirectional_phase_shift
        self.harmonic_scan = harmonic_scan
        self.laser_duty = laser_duty
        self.line_start_marker_delay = line_start_marker_delay
        self.line_stop_marker_delay = line_stop_marker_delay
        

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


    def to_dict(self):
        return {
            "lines": self.lines,
            "pixels": self.pixels,
            "frames": self.frames,
            "line_accumulations": self.line_accumulations,
            "bidirectional": self.bidirectional,
            "frame_start_marker": self.frame_start_marker_channel,
            "line_start_marker": self.line_start_marker_channel,
            "line_stop_marker": self.line_stop_marker_channel,
            "harmonic_scan": self.harmonic_scan,
            "laser_duty": self.laser_duty,
        }

    # TODO modify for number of sequences
    @classmethod
    def from_dict(cls, d):
        return cls(
            lines=d["lines"],
            pixels=d["pixels"],
            frames=d["frames"],
            line_accumulations=d.get("line_accumulations", (1,)),
            bidirectional=d.get("bidirectional", False),
            frame_start_marker_channel=d.get("frame_start_marker", 4),
            line_start_marker_channel=d.get("line_start_marker", 1),
            line_stop_marker_channel=d.get("line_stop_marker", 2),
            harmonic_scan=d.get("harmonic_scan", False),
            laser_duty=d.get("laser_duty", 0.0),
        )

    def __repr__(self):
        return f"ScanConfig({self.to_dict()})"


class ImageReconstructor:
    def __init__(
        self,
        config: ScanConfig,
        roi_mask: np.ndarray | None = None,
        outputs: Sequence[str] | None = None,
        laser_sync_rate: float = 40e6,
        tcspc_channels: int = 2**15,
        tcspc_bin_factor: int = 1,
        tcspc_resolution: float = 5e-12,
    ):
        """
        Initialize an image reconstructor.

        Parameters:
            config (ScanConfig): Scan configuration object.

            roi_mask (ndarray, optional): Binary mask of shape (lines, pixels)
                to restrict processing to specific regions.

            outputs (list of str, optional): List of outputs to compute.
                Each item must be one of the following:
                - "photon_count"
                - "mean_arrival_time"
                - "phasor"
                - "tcspc_histogram"
                If None, all outputs are computed.

            laser_sync_rate (float): Laser sync (repetition) rate in Hz,
                used together with tcspc_resolution to compute the angular
                frequency for phasor computation. Default: 40e6 (40 MHz).

            tcspc_channels (int): Number of time bins for TCSPC histogram.
                Cannot exceed 2**15, since dtime has 15-bit resolution.

            tcspc_bin_factor (int): Number of raw TCSPC channels to combine
                into a single histogram bin (e.g. 4 combines every 4 channels
                into 1), reducing resolution but also memory and noise for
                over-resolved acquisitions. Only affects tcspc_histogram;
                mean_arrival_time and phasor keep full raw dtime resolution.
                If it does not evenly divide tcspc_channels, tcspc_channels
                is rounded up to the nearest multiple. Default: 1 (no binning).

            tcspc_resolution (float): Raw TCSPC resolution in seconds per
                dtime channel (i.e. the file's MeasDesc_Resolution). Reported
                as-is in the output; used with tcspc_bin_factor to derive the
                binned tcspc_histogram_resolution. Default: 5e-12 (5 ps).
        """

        if not isinstance(config, ScanConfig):
            raise TypeError("ImageReconstructor requires a ScanConfig object")
        if not isinstance(tcspc_bin_factor, (int, np.integer)) or tcspc_bin_factor < 1:
            raise ValueError("tcspc_bin_factor must be a positive integer")
        max_tcspc_channels = 2**15  # dtime has 15-bit resolution
        if tcspc_channels > max_tcspc_channels:
            raise ValueError(
                f"tcspc_channels cannot exceed {max_tcspc_channels} "
                "(dtime has 15-bit resolution)"
            )
        # Round up to the nearest multiple of tcspc_bin_factor
        tcspc_channels = int(
            np.ceil(tcspc_channels / tcspc_bin_factor) * tcspc_bin_factor
        )
        self.config = config
        self.shape = (
            config.frames,
            config.lines * config._total_accumulations,
            config.pixels,
            config.max_detector,
        )
        # Maps a raw (pre-split) line index to its sequence index
        self._sequence_pattern = np.repeat(
            np.arange(len(config.line_accumulations)), config.line_accumulations
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

        # Initialize output arrays
        self.tcspc_channels = tcspc_channels
        self.tcspc_bin_factor = tcspc_bin_factor
        self.tcspc_hist_channels = tcspc_channels // tcspc_bin_factor
        # self.tcspc_resolution = tcspc_resolution
        self.tcspc_resolution = tcspc_resolution * tcspc_bin_factor
        self.omega = 2 * np.pi * laser_sync_rate * self.tcspc_resolution
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
                    len(self.config.line_accumulations),
                    self.config.max_detector,
                    self.tcspc_hist_channels,
                ),
                dtype=np.uint64,
            )  # existing shape logic

        # Rolling context
        self._partial_start_nsync = (
            None  # stores unmatched marker from previous chunk
        )
        self._current_line_idx = 0
        self._current_frame_idx = 0  # current frame index
        self._pending_photons = np.empty((0,), dtype=event_dtype)
        self._frame_marker_nsyncs = np.empty((0,), dtype=np.uint64)
 
        self._finished = False  # flag: stop processing when max frames reached

        self.stop_marker_phase = None
        self._stop_phase_computed = False
        self.line_duration = 0

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

        # Filter non-marker photons
        if len(self._pending_photons) > 0:
            events = np.concatenate([self._pending_photons, events])
            self._pending_photons = np.empty((0,), dtype=events.dtype)

        photons = get_photons(events)

        frame_markers, start_markers, stop_markers = resolve_markers(
            events,
            self.config.frame_start_marker_channel,
            self.config.line_start_marker_channel,
            self.config.line_stop_marker_channel,
        )
        if len(start_markers) == 0:
            # No line start markers → nothing to assemble this round
            return

        if not self._stop_phase_computed:
            self._compute_stop_phase(
                start_markers["nsync"], stop_markers["nsync"]
            )

        line_segments = self._build_line_segments(frame_markers, start_markers)

        self._assign_photons_to_segments(photons, line_segments)

    def finalize(self):
        if self._partial_start_nsync is not None:
            self._flush_final_line()

        data = {}
        active_detectors = sorted(self.active_detectors)
        channels = max(active_detectors) + 1

        data["tcspc_resolution"] = ((), self.tcspc_resolution)
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
                    len(self.config.line_accumulations),
                    self.config.lines,
                    self.config.pixels,
                    max(self.active_detectors) + 1,
                ),
                dtype=np.uint32,
            )

            pattern = np.repeat(
                np.arange(len(self.config.line_accumulations)),
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

            for accu_idx in range(len(self.config.line_accumulations)):
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
                # norm_phasor[photon_count == 0] = 0
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
            "sequence": np.arange(len(self.config.line_accumulations)),
            "line": np.arange(self.config.lines),
            "pixel": np.arange(self.config.pixels),
            "channel": np.arange(channels),
            "tcspc_time": np.arange(self.tcspc_hist_channels) * self.tcspc_resolution,
        }

        # Determine used dimensions
        used_dims = set()
        for data_array in data.values():
            dims = data_array[0]  # First item in tuple is dims
            used_dims.update(dims)

        # Only keep relevant coords
        coords = {k: v for k, v in all_coords.items() if k in used_dims}

        print("Reconstruction finished.")

        return xr.Dataset(data, coords=coords)

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

    @property
    def required_outputs(self):
        return list(self.requested_outputs)

    def get_available_outputs(self):
        return list(self.requested_outputs)

    def _stretch_roi_mask(self, base_mask: np.ndarray) -> np.ndarray:
        if base_mask.shape != (self.config.lines, self.config.pixels):
            raise ValueError(
                f"Base ROI mask must have shape ({self.config.lines}, {self.config.pixels})"
            )

        # Expand to (frames, sequences, lines, pixels)
        stretched_mask = np.zeros(
            (
                self.config.frames,
                self.config.lines * self.config._total_accumulations,
                self.config.pixels,
            ),
            dtype=bool,
        )

        for f_idx in range(self.config.frames):
            stretched_mask[f_idx, :, :] = np.repeat(
                base_mask, repeats=self.config._total_accumulations, axis=0
            )

        return stretched_mask
       
    def _build_line_segments(
        self,
        frame_markers: np.ndarray,
        start_markers: np.ndarray,
    ) -> np.ndarray:

        if self._finished:
            return np.empty(0, dtype=segment_dtype)

        frame_nsyncs = frame_markers["nsync"]
        start_nsyncs = start_markers["nsync"]

        self._frame_marker_nsyncs = np.append(
            self._frame_marker_nsyncs, frame_nsyncs, axis=0
        )

        if self._partial_start_nsync is not None:
            start_nsyncs = np.insert(
                start_nsyncs, 0, self._partial_start_nsync
            )
            self._partial_start_nsync = None

        self._partial_start_nsync = start_nsyncs[-1]

        start = start_nsyncs[:-1].astype(np.int64)
        stop = start + self.line_duration

        if len(start) == 0:
            return np.empty(0, dtype=segment_dtype)

        # Assign frames vectorized
        frame_idx = (
            np.searchsorted(self._frame_marker_nsyncs, start, side="right") - 1
        )

        # Clip to requested frame count
        valid = frame_idx < self.config.frames
        if not np.any(valid):
            self._finished = True
            return np.empty(0, dtype=segment_dtype)

        start = start[valid]
        stop = stop[valid]
        frame_idx = frame_idx[valid]

        # If bidirectional: even=forward, odd=reversed
        _, inverse, counts = np.unique(
            frame_idx, return_inverse=True, return_counts=True
        )
        cumsum = np.cumsum(np.r_[0, counts[:-1]])
        line_idx = np.arange(len(frame_idx)) - cumsum[inverse]

        if frame_idx[0] == self._current_frame_idx:
            line_idx[
                frame_idx == self._current_frame_idx
            ] += self._current_line_idx

        self._current_frame_idx = frame_idx[-1]
        self._current_line_idx = line_idx[-1] + 1

        reversed_mask = self.config.bidirectional & (line_idx % 2 == 1)

        # Delays shift the accepted photon window; unequal delays change its phase scale.
        start, stop = self._adjust_line_bounds(start, stop, reversed_mask)
   
        result = np.empty(len(start), dtype=segment_dtype)
        result["start_nsync"] = start
        result["stop_nsync"] = stop
        result["frame_idx"] = frame_idx
        result["line_idx"] = line_idx
        result["reversed"] = reversed_mask

        if not len(result):
            self._finished = True

        return result

    def _adjust_line_bounds(
        self,
        start: np.ndarray,
        stop: np.ndarray,
        reversed_flags: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return _adjust_line_bounds(
            start,
            stop,
            reversed_flags,
            self.line_duration,
            self.config.bidirectional,
            self.config.bidirectional_phase_shift,
            self.config.line_start_marker_delay,
            self.config.line_stop_marker_delay,
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

        if photons["dtime"].max() >= self.tcspc_channels:
            print(
                f"\033[91mWarning: TCSPC channel overflow detected. Max channel: {photons['dtime'].max()}\033[0m"
            )

        if photons["channel"].max() >= self.config.max_detector:
            print(
                f"\033[91mWarning: Channel overflow detected. Max channel: {photons['channel'].max()}\033[0m"
            )

        valid = (
            (segment_index >= 0)
            & (segment_index < len(segment_starts))
            & (photons["nsync"] < segment_ends[segment_index])
            & (photons["dtime"] < self.tcspc_channels)
            & (photons["channel"] < self.config.max_detector)
        )

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
            laser_duty=self.config.laser_duty,
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

        pending_photons_mask = photons["nsync"] >= segment_ends[-1]
        self._pending_photons = photons[pending_photons_mask]
        self.active_detectors.update(np.unique(channels))

    def _reshape_and_sum(
        self, array, f_idx, line_indices, lines, accum, pixels, channels
    ):
        sliced = array[f_idx, line_indices, :, :channels]
        reshaped = sliced.reshape(lines, accum, pixels, channels)
        return reshaped.sum(axis=1)

    def _flush_final_line(self):

        final_start = np.array([self._partial_start_nsync], dtype=np.int64)
        final_stop = final_start + self.line_duration
        final_reversed = np.array(
            [self.config.bidirectional and (self._current_line_idx % 2 == 1)],
            dtype=bool,
        )
        final_start, final_stop = self._adjust_line_bounds(
            final_start, final_stop, final_reversed
        )

        final_segment = np.empty(1, dtype=segment_dtype)
        final_segment["start_nsync"] = final_start
        final_segment["stop_nsync"] = final_stop
        final_segment["frame_idx"] = self._current_frame_idx
        final_segment["line_idx"] = self._current_line_idx
        final_segment["reversed"] = final_reversed

        self._assign_photons_to_segments(self._pending_photons, final_segment)
        self._pending_photons = np.empty(
            (0,), dtype=self._pending_photons.dtype
        )
        self._partial_start_nsync = None

    def _extract_markers(self, events, codes):
        return get_markers(events, codes)

    def _compute_stop_phase(
        self,
        start_nsyncs: NDArray[np.uint64],
        stop_nsyncs: NDArray[np.uint64],
        default_phase: float = 0.80,
    ) -> None:
        if start_nsyncs.dtype != np.uint64 or stop_nsyncs.dtype != np.uint64:
            raise TypeError(
                "start_nsyncs and stop_nsyncs must be uint64 arrays"
            )
        start_nsyncs = start_nsyncs.astype(np.int64, copy=False)
        stop_nsyncs = stop_nsyncs.astype(np.int64, copy=False)

        if len(start_nsyncs) < 2:
            print("No valid start markers. Phase not calculated!")
            self.stop_marker_phase = None
            self.line_duration = 0
            return

        if len(stop_nsyncs) == 0:
            print("No valid stop markers. Using default!")
            intervals = start_nsyncs[1:] - start_nsyncs[:-1]
            self.stop_marker_phase = default_phase
            self.line_duration = int(np.median(intervals) * default_phase)
            self._stop_phase_computed = True
            return

        # Paired analysis
        pair_count = min(len(stop_nsyncs), len(start_nsyncs) - 1)
        durations = stop_nsyncs[:pair_count] - start_nsyncs[:pair_count]
        intervals = (
            start_nsyncs[1 : 1 + pair_count] - start_nsyncs[:pair_count]
        )

        # Basic sanity check
        if np.any(durations <= 0):
            print("Invalid stop markers (<= start). Using default.")
            intervals = start_nsyncs[1:] - start_nsyncs[:-1]
            self.stop_marker_phase = default_phase
            self.line_duration = int(np.median(intervals) * default_phase)
            self._stop_phase_computed = True
            return

        phase_estimates = durations / intervals
        valid = (phase_estimates > 0) & (phase_estimates < 1)

        if not np.any(valid):
            print("No valid stop marker timings. Using default.")
            intervals = start_nsyncs[1:] - start_nsyncs[:-1]
            self.stop_marker_phase = default_phase
            self.line_duration = int(np.median(intervals) * default_phase)
        else:
            self.stop_marker_phase = float(np.median(phase_estimates[valid]))
            self.line_duration = int(np.median(durations[valid]))

        self._stop_phase_computed = True
        return

# TODO: add option to select marker channels and validation of marker chan


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
    """

    def __init__(self, config: ScanConfig):
        if not isinstance(config, ScanConfig):
            raise TypeError("SegmentReconstructor requires a ScanConfig object")
        self.config = config
        self.stop_marker_phase = None
        self.line_duration = 0

    def reconstruct(self, events: np.ndarray) -> xr.Dataset:
        """Process one chunk of events and return the reconstructed image.

        Args:
            events: Array of events with dtype ``event_dtype``.

        Returns:
            xr.Dataset with a ``photon_count`` variable of dims
            ``("line", "pixel")`` and a ``frame_break_line`` variable listing
            the line index after which each frame-start marker found in the
            chunk occurred (empty if the chunk doesn't span a frame boundary).
        """
        if events.dtype != event_dtype:
            raise TypeError(
                f"Expected events with dtype {event_dtype}, got {events.dtype}"
            )

        photons = get_photons(events)
        frame_markers, start_markers, stop_markers = resolve_markers(
            events,
            self.config.frame_start_marker_channel,
            self.config.line_start_marker_channel,
            self.config.line_stop_marker_channel,
        )

        if len(start_markers) < 2:
            # Not enough line-start markers to bound a single complete line
            return self._empty_dataset()

        # Drop stop markers belonging to a line whose start was in a previous
        # chunk (i.e. preceding this chunk's first start marker), so
        # stop_markers[0] truly pairs with start_markers[0].
        stop_markers = stop_markers[stop_markers["nsync"] > start_markers["nsync"][0]]

        self._compute_stop_phase(start_markers["nsync"], stop_markers["nsync"])
        if self.line_duration <= 0:
            return self._empty_dataset()

        start, stop, line_idx, reversed_mask = self._build_segments(
            start_markers["nsync"]
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
            },
            coords={
                "line": np.arange(len(start)),
                "pixel": np.arange(self.config.pixels),
                "frame_break": np.arange(len(frame_break_lines)),
            },
        )

    def _find_frame_breaks(
        self, frame_nsyncs: NDArray[np.uint64], start_nsyncs: NDArray[np.uint64]
    ) -> np.ndarray:
        """For each frame-start marker, find the index of the line right
        before it (i.e. the last line of the previous frame)."""
        if len(frame_nsyncs) == 0:
            return np.empty(0, dtype=np.int64)

        used_starts = start_nsyncs[:-1].astype(np.int64)
        idx = np.searchsorted(used_starts, frame_nsyncs.astype(np.int64), side="right") - 1
        # A frame marker before the first reconstructed line has no previous
        # line to report a break after; drop it instead of clamping to 0.
        # idx = idx[idx >= 0]
        return np.clip(idx, 0, len(used_starts) - 1)

    def _empty_dataset(self) -> xr.Dataset:
        return xr.Dataset(
            {
                "photon_count": (
                    ("line", "pixel"),
                    np.zeros((0, self.config.pixels), dtype=np.uint32),
                ),
                "frame_break_line": (("frame_break",), np.empty(0, dtype=np.int64)),
            },
            coords={
                "line": np.arange(0),
                "pixel": np.arange(self.config.pixels),
                "frame_break": np.arange(0),
            },
        )

    def _compute_stop_phase(
        self,
        start_nsyncs: NDArray[np.uint64],
        stop_nsyncs: NDArray[np.uint64],
        default_phase: float = 0.80,
    ) -> None:
        start_nsyncs = start_nsyncs.astype(np.int64, copy=False)
        stop_nsyncs = stop_nsyncs.astype(np.int64, copy=False)

        if len(stop_nsyncs) == 0:
            intervals = start_nsyncs[1:] - start_nsyncs[:-1]
            self.stop_marker_phase = default_phase
            self.line_duration = int(np.median(intervals) * default_phase)
            return

        # Use only the first start/stop pair, as requested
        start0, stop0 = start_nsyncs[0], stop_nsyncs[0]
        interval0 = start_nsyncs[1] - start_nsyncs[0]
        duration0 = stop0 - start0

        if duration0 <= 0 or not (0 < duration0 / interval0 < 1):
            intervals = start_nsyncs[1:] - start_nsyncs[:-1]
            self.stop_marker_phase = default_phase
            self.line_duration = int(np.median(intervals) * default_phase)
            return

        self.stop_marker_phase = float(duration0 / interval0)
        self.line_duration = int(duration0)

    def _build_segments(
        self, start_nsyncs: NDArray[np.uint64]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        start = start_nsyncs[:-1].astype(np.int64)
        stop = start + self.line_duration
        line_idx = np.arange(len(start))
        reversed_mask = self.config.bidirectional & (line_idx % 2 == 1)

        start, stop = _adjust_line_bounds(
            start,
            stop,
            reversed_mask,
            self.line_duration,
            self.config.bidirectional,
            self.config.bidirectional_phase_shift,
            self.config.line_start_marker_delay,
            self.config.line_stop_marker_delay,
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
        start_time: float = 0.0,
        stop_time: float = 10.0,
        max_detector: int = 64,
        bin_width: float = 1e-3,
        sync_rate: float = 40e6,
        outputs: Sequence[str] | None = None,
    ):
        """
        Initialize a trace reconstructor.

        Parameters:
            start_time_s (float): Start time in seconds. Default: 0.0
            stop_time_s (float): Stop time in seconds. Default: 10.0
            max_detector (int): Maximum detector index. Default: 64
            bin_width_s (float): Time bin width in seconds. Default: 1e-3 (1 ms)
            sync_rate (float): Sync rate in Hz. Default: 40e6 (40 MHz)
            outputs (list of str, optional): List of outputs to compute.
                - "photon_count": intensity trace per detector
                - "markers": marker times (frame start, line start, line stop)
                If None, both outputs are computed.
        """

        if sync_rate <= 0:
            raise ValueError("sync_rate must be positive")
        if bin_width <= 0:
            raise ValueError("bin_width_s must be positive")
        if start_time >= stop_time:
            raise ValueError("start_time_s must be less than stop_time_s")

        self.start_time = start_time
        self.stop_time = stop_time
        self.max_detector = max_detector
        self.bin_width = bin_width
        self.sync_rate = sync_rate

        # Calculate number of bins
        self.n_bins = int(np.ceil((stop_time - start_time) / bin_width))

        # Convert time boundaries to nsync
        self.start_nsync = int(start_time * sync_rate)
        self.stop_nsync = int(stop_time * sync_rate)
        self.bin_width_nsync = int(bin_width * sync_rate)

        # Create time edges and bin centers
        self.time_edges_nsync = (
            np.arange(self.n_bins + 1, dtype=np.float64) * self.bin_width_nsync
        ) + self.start_nsync
        self.time_axis = (
            (np.arange(self.n_bins, dtype=np.float64) + 0.5) * self.bin_width
            + self.start_time
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
                (self.n_bins, max_detector), dtype=np.uint64
            )

        if "markers" in self.requested_outputs:
            self.marker_times = {
                "frame_start": np.array([], dtype=np.uint64),
                "line_start": np.array([], dtype=np.uint64),
                "line_stop": np.array([], dtype=np.uint64),
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
            photons = get_photons(events)

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
                & (photons["channel"] < self.max_detector)
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
            (
                frame_markers,
                line_start_markers,
                line_stop_markers,
            ) = resolve_markers(events, 4, 1, 2)

            # Filter markers within time range
            frame_valid = (
                (frame_markers["nsync"] >= self.start_nsync)
                & (frame_markers["nsync"] < self.stop_nsync)
            )
            line_start_valid = (
                (line_start_markers["nsync"] >= self.start_nsync)
                & (line_start_markers["nsync"] < self.stop_nsync)
            )
            line_stop_valid = (
                (line_stop_markers["nsync"] >= self.start_nsync)
                & (line_stop_markers["nsync"] < self.stop_nsync)
            )

            self.marker_times["frame_start"] = np.append(
                self.marker_times["frame_start"],
                frame_markers["nsync"][frame_valid],
            )
            self.marker_times["line_start"] = np.append(
                self.marker_times["line_start"],
                line_start_markers["nsync"][line_start_valid],
            )
            self.marker_times["line_stop"] = np.append(
                self.marker_times["line_stop"],
                line_stop_markers["nsync"][line_stop_valid],
            )

    def finalize(self) -> xr.Dataset:
        """Finalize and return results as xarray Dataset.
        
        Returns:
            xr.Dataset: Dataset containing:
                - photon_count: intensity trace with dimensions (time, channel)
                - frame_start_times: frame start marker times
                - line_start_times: line start marker times
                - line_stop_times: line stop marker times
        """

        data = {}
        channels = max(self.active_detectors) + 1 if self.active_detectors else 1

        if "photon_count" in self.requested_outputs:
            # Trim intensity array to active channels
            intensity = self.intensity[:, :channels]
            data["photon_count"] = (("time", "channel"), intensity)

        if "markers" in self.requested_outputs:
            # Convert marker times from nsync to seconds
            frame_start = (
                self.marker_times["frame_start"] / self.sync_rate
            )
            line_start = (
                self.marker_times["line_start"] / self.sync_rate
            )
            line_stop = (
                self.marker_times["line_stop"] / self.sync_rate
            )

            # Always add marker fields, even if empty
            data["frame_start_times"] = (("frame_marker",), frame_start)
            data["line_start_times"] = (("line_start_marker",), line_start)
            data["line_stop_times"] = (("line_stop_marker",), line_stop)

        coords = {
            "time": self.time_axis,
            "channel": np.arange(channels),
        }

        if "markers" in self.requested_outputs:
            # Always add marker coordinates, even if empty
            coords["frame_marker"] = np.arange(
                len(self.marker_times["frame_start"])
            )
            coords["line_start_marker"] = np.arange(
                len(self.marker_times["line_start"])
            )
            coords["line_stop_marker"] = np.arange(
                len(self.marker_times["line_stop"])
            )

        return xr.Dataset(data, coords=coords)



