"""Reading and image reconstruction of PicoQuant T3 (.ptu) files.

The usual workflow::

    from tttrkit.ptuio import (
        ImageReconstructor, ScanConfig, T3OverflowCorrector, TTTRReader,
        prescan_first_frame,
    )

    reader = TTTRReader(path)
    config = ScanConfig(lines=512, pixels=512)
    prescan_first_frame(reader, config)  # checks config, fills line_duration
    rec = ImageReconstructor(
        config,
        sync_rate=reader.header.sync_rate,
        tcspc_resolution=reader.header.tcspc_resolution,
    )
    corrector = T3OverflowCorrector()
    for chunk in reader.iter_chunks():
        rec.update(corrector.correct(chunk))
    ds = rec.finalize()

Estimation from the data (TCSPC channels, bidirectional alignment) is in
:mod:`tttrkit.ptuio.calibration`, marker diagnostics in
:mod:`tttrkit.ptuio.diagnostics`; lower-level pieces (record decoding,
marker timing) in their own submodules.
"""

from .config import (
    DataQualityWarning,
    FrameSyncWarning,
    ScanConfig,
    ScanConfigWarning,
)
from .decoder import T3OverflowCorrector
from .prescan import FirstFrameScan, prescan_events, prescan_first_frame
from .reader import TTTRReader
from .reconstructor import (
    ImageReconstructor,
    SegmentReconstructor,
    TraceReconstructor,
)

__all__ = [
    "DataQualityWarning",
    "FirstFrameScan",
    "FrameSyncWarning",
    "ImageReconstructor",
    "ScanConfig",
    "ScanConfigWarning",
    "SegmentReconstructor",
    "T3OverflowCorrector",
    "TTTRReader",
    "TraceReconstructor",
    "prescan_events",
    "prescan_first_frame",
]
