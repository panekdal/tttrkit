"""ScanConfig: derived values follow their inputs; assignments are validated."""

import numpy as np
import pytest

from tttrkit.ptuio.config import ScanConfig
from tttrkit.ptuio.reconstructor import ImageReconstructor

from test_frame_sync import ACQUISITION

GEOMETRY = dict(lines=4, pixels=4)


def test_derived_values_follow_line_accumulations():
    config = ScanConfig(lines=512, pixels=4, line_accumulations=np.int64(2))
    assert config.line_accumulations == (2,)
    config.line_accumulations = [1, 3]
    assert config.line_accumulations == (1, 3)
    assert config.num_sequences == 2
    assert config.total_accumulations == 4
    assert config.lines_per_frame == 2048
    assert config.frame_sync_tolerance_lines == 204


@pytest.mark.parametrize(
    "name, value, error",
    [
        ("frame_sync", "typo", ValueError),
        ("frame_sync_tolerance", 0.5, ValueError),
        ("line_accumulations", (), ValueError),
        ("line_accumulations", (1, 0), ValueError),
        ("frame_start_marker_channel", 3, ValueError),
        ("line_start_marker_channel", 1.0, TypeError),
        ("scan_duty", 0.0, ValueError),
        ("scan_duty", 1.5, ValueError),
    ],
)
def test_assignments_are_validated(name, value, error):
    with pytest.raises(error):
        ScanConfig(**GEOMETRY, **{name: value})
    config = ScanConfig(**GEOMETRY)
    with pytest.raises(error):
        setattr(config, name, value)


@pytest.mark.parametrize("missing", ["lines", "pixels"])
def test_geometry_is_required(missing):
    geometry = {k: v for k, v in GEOMETRY.items() if k != missing}
    with pytest.raises(TypeError, match=missing):
        ScanConfig(**geometry)
    with pytest.raises(TypeError, match=missing):
        ScanConfig.from_dict(geometry)


def test_harmonic_scan_requires_laser_duty():
    with pytest.raises(ValueError, match="scan_duty"):
        ScanConfig(**GEOMETRY, harmonic_scan=True)
    assert ScanConfig(**GEOMETRY, harmonic_scan=True, scan_duty=0.6).scan_duty == 0.6
    # Switched on later: refused by the reconstructor
    config = ScanConfig(**GEOMETRY, line_duration=100)
    config.harmonic_scan = True
    with pytest.raises(ValueError, match="scan_duty"):
        ImageReconstructor(config, **ACQUISITION)


def test_round_trip_through_dict():
    config = ScanConfig(**GEOMETRY, line_accumulations=(1, 2), frame_sync="count")
    assert ScanConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(TypeError, match="lnies"):
        ScanConfig.from_dict({**GEOMETRY, "lnies": 4})


def test_image_reconstructor_keeps_its_own_config():
    config = ScanConfig(**GEOMETRY, line_duration=100)
    rec = ImageReconstructor(
        config, outputs=["photon_count"], tcspc_channels=16, **ACQUISITION
    )
    config.pixels = 8
    assert rec.config.pixels == 4


@pytest.mark.parametrize("missing", ["sync_rate", "tcspc_resolution"])
def test_image_reconstructor_requires_acquisition_values(missing):
    config = ScanConfig(**GEOMETRY, line_duration=100)
    given = {k: v for k, v in ACQUISITION.items() if k != missing}
    with pytest.raises(TypeError, match=missing):
        ImageReconstructor(config, **given)


def test_histogram_binning_affects_only_the_histogram():
    from tttrkit.ptuio.prescan import prescan_events
    from test_frame_sync import make_stream

    events = make_stream()

    def run(factor):
        config = ScanConfig(lines=8, pixels=4, frames=3, max_detector_channels=2)
        prescan_events([events], config, warn=False)
        rec = ImageReconstructor(
            config, tcspc_channels=1024, tcspc_bin_factor=factor, **ACQUISITION
        )
        rec.update(events)
        return rec.finalize()

    raw, binned = run(1), run(4)
    assert binned.tcspc_resolution == raw.tcspc_resolution == ACQUISITION["tcspc_resolution"]
    assert binned.tcspc_histogram_resolution == 4 * raw.tcspc_histogram_resolution
    assert binned.omega == raw.omega
    np.testing.assert_allclose(binned.phasor_g, raw.phasor_g)
    np.testing.assert_allclose(binned.phasor_s, raw.phasor_s)
    np.testing.assert_array_equal(binned.mean_arrival_time, raw.mean_arrival_time)
    assert binned.sizes["tcspc_time"] * 4 == raw.sizes["tcspc_time"]
