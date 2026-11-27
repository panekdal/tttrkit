"""tttrkit as used by napari-flopa, on the sample sets.

napari-flopa builds its ScanConfig from its own saved scan configuration and
drives ImageReconstructor itself, so changes in tttrkit's API (ScanConfig
keywords, attributes it reads back) break it without failing tttrkit's own
tests. For every sample file with a napari-flopa sidecar
(<name>_scan_config.json), this runs napari-flopa's loading and
reconstruction pipeline and checks that it gives the same photon counts as
the regression harness, which reads the same sidecar without napari-flopa.

Needs napari-flopa importable, e.g. run from napari-flopa's environment with
this checkout installed in editable mode:

    ..\\napari-flopa\\.venv\\Scripts\\python -m pytest tests/test_napari_flopa.py
"""

import contextlib
import io

import numpy as np
import pytest

import realdata_harness as h

flopa = pytest.importorskip("napari_flopa")

from napari_flopa.core.io.config import load_config, scan_config_from_dict  # noqa: E402
from napari_flopa.core.io.loader import read_ptu_file  # noqa: E402
from napari_flopa.core.logger import ProgressLogger  # noqa: E402
from napari_flopa.core.processing.reconstruction import (  # noqa: E402
    reconstruct_ptu_to_dataset,
)

from tttrkit.ptuio.reconstructor import ImageReconstructor, ScanConfig  # noqa: E402

FULL_TCSPC = 2**15


def _sample_files():
    if not h.SAMPLE_DIR.exists():
        return []
    return [
        p for p in h.find_ptus(h.SAMPLE_DIR)
        if p.with_name(p.stem + h.SIDECAR_SUFFIX).exists()
    ]


@pytest.mark.realdata
@pytest.mark.parametrize(
    "ptu", _sample_files(), ids=lambda p: p.relative_to(h.SAMPLE_DIR).as_posix()
)
def test_napari_flopa_pipeline(ptu):
    sidecar = ptu.with_name(ptu.stem + h.SIDECAR_SUFFIX)
    config, recon_kwargs = h.sidecar_config(ptu)
    tcspc = recon_kwargs.get("tcspc_channels", FULL_TCSPC)

    with contextlib.redirect_stdout(io.StringIO()):
        # tttrkit directly, as the regression harness reads the sidecar
        reader, corr = h._open(ptu)
        cfg = ScanConfig(**config)
        h.prescan(reader, cfg, h.CHUNK_SIZE)
        rec = ImageReconstructor(
            cfg,
            outputs=["photon_count"],
            tcspc_channels=tcspc,
            sync_rate=reader.header.sync_rate,
            tcspc_resolution=reader.header.tcspc_resolution,
        )
        for chunk in reader.iter_chunks(h.CHUNK_SIZE):
            rec.update(corr.correct(chunk))
        direct = rec.finalize()

        # napari-flopa: its own config loader and pipeline
        ptu_data = read_ptu_file(str(ptu), header=False, logger=ProgressLogger("collect"))
        rate = ptu_data["constants"]["repetition_rate"]
        run = lambda: reconstruct_ptu_to_dataset(  # noqa: E731
            ptu_data,
            scan_config_from_dict(load_config(sidecar), rate),
            outputs=["photon_count"],
            tcspc_channels_override=tcspc,
            logger=ProgressLogger("collect"),
        )
        if not direct["photon_count"].values.any():
            # napari-flopa refuses empty reconstructions
            with pytest.raises(RuntimeError, match="No photons"):
                run()
            return
        ds = run()

    np.testing.assert_array_equal(
        ds["photon_count"].values, direct["photon_count"].values
    )
    # napari-flopa reports the line duration tttrkit measured
    if config.get("line_duration") is None and rec.line_duration:
        assert ds.attrs["line_duration_s"] == pytest.approx(rec.line_duration / rate)
