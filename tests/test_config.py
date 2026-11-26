"""ScanConfig: derived values follow their inputs; assignments are validated."""

import numpy as np
import pytest

from tttrkit.ptuio.config import ScanConfig
from tttrkit.ptuio.reconstructor import ImageReconstructor


def test_derived_values_follow_line_accumulations():
    config = ScanConfig(lines=512, line_accumulations=np.int64(2))
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
    ],
)
def test_assignments_are_validated(name, value, error):
    with pytest.raises(error):
        ScanConfig(**{name: value})
    config = ScanConfig()
    with pytest.raises(error):
        setattr(config, name, value)


def test_round_trip_through_dict():
    config = ScanConfig(lines=4, line_accumulations=(1, 2), frame_sync="count")
    assert ScanConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()


def test_image_reconstructor_keeps_its_own_config():
    config = ScanConfig(lines=4, pixels=4, line_duration=100)
    rec = ImageReconstructor(config, outputs=["photon_count"], tcspc_channels=16)
    config.pixels = 8
    assert rec.config.pixels == 4
