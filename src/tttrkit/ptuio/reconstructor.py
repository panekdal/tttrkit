from collections.abc import Sequence
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from .decoder import event_dtype, get_markers, get_photons

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


class ScanConfig:
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
        harmonic_phase: float = 0.0,
    ):
        self.lines = lines
        self.pixels = pixels
        self.frames = frames
        self.max_detector = max_detector
        self.bidirectional_phase_shift = bidirectional_phase_shift
        self.harmonic_scan = harmonic_scan
        self.harmonic_phase = harmonic_phase
        

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
            "harmonic_phase": self.harmonic_phase,
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
            harmonic_phase=d.get("harmonic_phase", 0.0),
        )

    def __repr__(self):
        return f"ScanConfig({self.to_dict()})"


class ImageReconstructor:
    def __init__(
        self,
        config: ScanConfig,
        roi_mask: np.ndarray | None = None,
        outputs: Sequence[str] | None = None,
        omega: float = 0.012,
        tcspc_channels: int = 2**15,
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

            omega (float): Angular frequency for phasor computation.

            tcspc_channels (int): Number of time bins for TCSPC histogram.
        """

        if not isinstance(config, ScanConfig):
            raise TypeError("ImageReconstructor requires a ScanConfig object")
        self.config = config
        self.shape = (
            config.frames,
            config.lines * config._total_accumulations,
            config.pixels,
            config.max_detector,
        )
        self.omega = omega
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
        if "arrival_sum" in self._required:
            self.arrival_sum = np.zeros(self.shape, dtype=np.float32)
        if "photon_count" in self._required:
            self.photon_count = np.zeros(self.shape, dtype=np.uint32)
        if "phasor_sum" in self._required:
            self.phasor_sum = np.zeros(self.shape, dtype=np.complex64)
        if "tcspc_hist" in self._required:
            self.tcspc_hist = np.zeros(
                (self.config.frames, self.config.max_detector, tcspc_channels),
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
        # self._start_marker_nsyncs = np.empty((0,), dtype=np.ndarray)
        # self._stop_marker_nsyncs = np.empty((0,), dtype=np.ndarray)

        self._finished = False  # flag: stop processing when max frames reached

        self.stop_marker_phase = None
        self._stop_phase_computed = False
        self.line_duration = 0

        # ROI for masking
        self._roi_mask_stretched = None
        if roi_mask is not None:
            self._roi_mask_stretched = self._stretch_roi_mask(roi_mask)
        
        # Build harmonic correction lookup table
        # self._harmonic_lut = None
        # if self.config.harmonic_scan:
        #     self._harmonic_lut = self._build_harmonic_lut()

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

        # Extract frame markers
        frame_markers = get_markers(
            events, self.config.frame_start_marker_channel
        )

        # Extract line markers
        start_markers = get_markers(
            events, self.config.line_start_marker_channel
        )
        if len(start_markers) == 0:
            # No line start markers → nothing to assemble this round
            return

        stop_markers = (
            get_markers(events, self.config.line_stop_marker_channel)
            if self.config.line_stop_marker_channel
            else None
        )

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

        if "tcspc_histogram" in self.requested_outputs:
            self.tcspc_hist = self.tcspc_hist[:, :channels, :]
            data["tcspc_histogram"] = (
                ("frame", "channel", "tcspc_channel"),
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
            "tcspc_channel": np.arange(self.tcspc_channels),
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

    def _harmonic_correction(self, temporal_phase: np.ndarray) -> np.ndarray:
        phi = self.config.harmonic_phase
        I = 2 * np.sin(0.5 * np.pi * phi) / (np.pi * phi)
        y = ( np.sin(np.pi * phi / 2)  - np.cos(np.pi * phi * temporal_phase + 0.5 * np.pi * (1-phi))) / I / np.pi / phi
        return y


        # velocity = np.sin(np.pi * phi * temporal_phase + 0.5 * np.pi * (1-phi))

        # phase_corrected = temporal_phase / velocity
        # # phase_corrected = temporal_phase
        # phase_corrected = 0.5 + np.arcsin(
        #     (2 * temporal_phase - 1) * np.sin(0.5 * np.pi * phi)
        # ) / (np.pi * phi)

        # y = np.asarray(y, dtype=float)

        # s = np.sin(0.5 * np.pi * phi)
        # return 0.5 + np.arcsin((2 * temporal_phase - 1) * s) / (np.pi * phi)



        # # Map temporal phase [0, 1] to absolute angle [φ, φ + 2π]
        # angle = 2 * np.pi * temporal_phase + phi
        
        # # Convert scanner position (sin range [-1,1]) to pixel range [0,1]
        # position = (np.sin(angle) + 1.0) / 2.0
        
        # return phase_corrected
    
    # def _build_harmonic_lut(self) -> np.ndarray:
    #     """
    #     Build lookup table for harmonic phase correction.
    #     Maps temporal phase indices to corrected pixel phases.
        
    #     Returns:
    #         Array of shape (pixels + 1,) with correction values
    #     """
    #     temporal_phases = np.linspace(0, 1, self.config.pixels + 1)
    #     pixel_phases = self._harmonic_correction(temporal_phases)
    #     return pixel_phases
    
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

        if self.config.bidirectional:
            shift = int(
                self.config.bidirectional_phase_shift * self.line_duration
            )
            start[reversed_mask] += shift
            stop[reversed_mask] += shift

        result = np.empty(len(start), dtype=segment_dtype)
        result["start_nsync"] = start
        result["stop_nsync"] = stop
        result["frame_idx"] = frame_idx
        result["line_idx"] = line_idx
        result["reversed"] = reversed_mask

        if not len(result):
            self._finished = True

        return result

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

        phase = (
            photons_in_segments["nsync"].astype(np.int64)
            - segment_starts[segment_index]
        ) / (segment_ends[segment_index] - segment_starts[segment_index])
        
        # Apply harmonic scan correction if enabled
        if self.config.harmonic_scan:
            phase = self._harmonic_correction(phase)

        pixels = np.floor(phase * self.config.pixels).astype(int)

        pixels = np.where(
            reversed_flags, self.config.pixels - 1 - pixels, pixels
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
        phasors = np.exp(1j * self.omega * dtimes)

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
            np.add.at(self.tcspc_hist, (frames, channels, dtimes), 1)

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

        final_segment = np.empty(1, dtype=segment_dtype)
        final_segment["start_nsync"] = self._partial_start_nsync
        final_segment["stop_nsync"] = (
            self._partial_start_nsync + self.line_duration
        )
        final_segment["frame_idx"] = self._current_frame_idx
        final_segment["line_idx"] = self._current_line_idx
        final_segment["reversed"] = self.config.bidirectional and (
            self._current_line_idx % 2 == 1
        )

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
            frame_markers = get_markers(events, 4)  # Frame start = mask 4
            line_start_markers = get_markers(events, 1)  # Line start = mask 1
            line_stop_markers = get_markers(events, 2)  # Line stop = mask 2

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



