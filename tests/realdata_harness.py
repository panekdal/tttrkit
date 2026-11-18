"""Real-data regression harness for ImageReconstructor.

Every .ptu file under a data directory is reconstructed and fingerprinted
(SHA-256 of each output array plus line/column/TCSPC profiles for
diagnostics). Fingerprints are compared with stored references, so any
change in reconstruction behaviour on real data is detected and has to be
accepted explicitly by regenerating the references.

Default data: the sample sets in tests/realdata_samples (git-ignored,
local to this machine):
    clean/        files without acquisition anomalies, one or more per system
    anomalies/    files with marker glitches and other edge cases
    fingerprints/ their reference fingerprints
Any other directory (e.g. the whole data folder) can be tested too; its
fingerprints go to tests/realdata_samples/corpus_fingerprints/.

Scan configuration per file, first match wins:
    1. <name>_scan_config.json   napari-flopa scan configuration next to the file
    2. stored reference          config inferred when the reference was created
    3. inferred                  from header (ImgHdr_PixY/PixX) and marker counts
tests/realdata_cases.json can add variants of a sample file (config
overrides, e.g. frame_sync), chunk sizes to compare and expectations.

Used by tests/test_realdata.py; can also be run directly:
    python tests/realdata_harness.py [DATA_DIR] [--update | --add-missing]
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import io
import json
import os
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SAMPLE_DIR = Path(__file__).with_name("realdata_samples")
MANIFEST = Path(__file__).with_name("realdata_cases.json")
SIDECAR_SUFFIX = "_scan_config.json"
CHUNK_SIZE = 1_000_000
MAX_DETECTOR_CAP = 8  # bounds memory for inferred configs
DEFAULT_WORKERS = min(4, os.cpu_count() or 1)


class SkipCase(Exception):
    """File cannot be reconstructed as an image (e.g. no line markers)."""


@dataclass
class Case:
    id: str
    path: str  # relative to the data directory, forward slashes
    source: str  # "sidecar", "reference" or "inferred"
    config: dict | None = None  # ScanConfig kwargs; None -> infer
    reconstructor: dict = field(default_factory=dict)
    overrides: dict = field(default_factory=dict)  # applied on top of config
    chunk_sizes: list = field(default_factory=lambda: [CHUNK_SIZE])
    expect: dict = field(default_factory=dict)

    def reference_path(self, refs_dir: Path) -> Path:
        return refs_dir / f"{self.id}.npz"


# ----------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------
def is_sample_dir(data_dir: Path) -> bool:
    return Path(data_dir).resolve() == SAMPLE_DIR.resolve()


def default_refs(data_dir: Path) -> Path:
    if is_sample_dir(data_dir):
        return SAMPLE_DIR / "fingerprints"
    return SAMPLE_DIR / "corpus_fingerprints"


def find_ptus(data_dir: Path) -> list[Path]:
    ptus = []
    for root, _dirs, files in os.walk(data_dir):
        ptus += [Path(root) / f for f in files if f.lower().endswith(".ptu")]
    return sorted(ptus)


def load_manifest(path: Path = MANIFEST) -> dict[str, list[dict]]:
    """Sample-file path -> list of case entries."""
    if not path.exists():
        return {}
    by_path = {}
    for entry in json.loads(path.read_text(encoding="utf-8"))["cases"]:
        by_path.setdefault(entry["path"], []).append(entry)
    return by_path


def sidecar_config(ptu: Path) -> tuple[dict, dict] | None:
    """(ScanConfig kwargs, reconstructor kwargs) from a napari-flopa sidecar.

    napari-flopa stores marker delays and the line duration in seconds;
    ScanConfig takes sync counts. Sidecars written before that change
    (recognised by bidirectional_phase_shift) stored delays as fractions of
    the line duration, which cannot be converted; their delays are dropped.
    """
    sidecar = ptu.with_name(ptu.stem + SIDECAR_SUFFIX)
    if not sidecar.exists():
        return None
    scan = dict(json.loads(sidecar.read_text(encoding="utf-8"))["scan"])
    seconds = ("line_start_marker_delay", "line_stop_marker_delay", "line_duration")
    if "bidirectional_phase_shift" in scan:
        for key in seconds + ("bidirectional_phase_shift",):
            scan.pop(key, None)
    elif any(scan.get(k) is not None for k in seconds):
        reader, _ = _open(ptu)
        sync_rate = float(reader.header.tags["TTResult_SyncRate"])
        for key in seconds:
            if scan.get(key) is not None:
                scan[key] = int(round(float(scan[key]) * sync_rate))
    rename = {
        "accumulations": "line_accumulations",
        "frame_start_marker": "frame_start_marker_channel",
        "line_start_marker": "line_start_marker_channel",
        "line_stop_marker": "line_stop_marker_channel",
    }
    skip = {"sequences", "tcspc_bins"}
    config = {rename.get(k, k): v for k, v in scan.items() if k not in skip}
    recon = {}
    if "tcspc_bins" in scan:
        recon["tcspc_channels"] = int(scan["tcspc_bins"])
    return config, recon


def discover_cases(data_dir: Path, refs_dir: Path | None = None) -> list[Case]:
    data_dir = Path(data_dir)
    refs_dir = Path(refs_dir) if refs_dir else default_refs(data_dir)
    manifest = load_manifest() if is_sample_dir(data_dir) else {}

    cases = []
    for ptu in find_ptus(data_dir):
        rel = ptu.relative_to(data_dir).as_posix()
        base = Case(id=rel, path=rel, source="inferred")
        side = sidecar_config(ptu)
        if side is not None:
            base.source = "sidecar"
            base.config, base.reconstructor = side
        else:
            ref = base.reference_path(refs_dir)
            if ref.exists():
                meta = load_reference(ref)["meta"]
                if meta.get("config_source") == "inferred":
                    base.source = "reference"
                    base.config = meta["config"]
                    base.reconstructor = meta["reconstructor"]
        for entry in manifest.get(rel, [{}]):
            cases.append(
                Case(
                    id=entry.get("id", rel),
                    path=rel,
                    source=base.source,
                    config=base.config,
                    reconstructor=dict(base.reconstructor),
                    overrides=entry.get("config", {}),
                    chunk_sizes=entry.get("chunk_sizes", [CHUNK_SIZE]),
                    expect=entry.get("expect", {}),
                )
            )
    return cases


# ----------------------------------------------------------------------
# Reconstruction and fingerprint
# ----------------------------------------------------------------------
def _supported_kwargs(func, kwargs: dict) -> dict:
    """Drop kwargs an older version of the API does not accept."""
    params = inspect.signature(func).parameters
    return {k: v for k, v in kwargs.items() if k in params}


def _open(path: Path):
    from tttrkit.ptuio.decoder import T3OverflowCorrector
    from tttrkit.ptuio.reader import TTTRReader

    reader = TTTRReader(str(path))
    wrap = reader.header.tags.get("TTResultFormat_WrapAround", 1024)
    return reader, T3OverflowCorrector(wraparound=wrap)


def infer_config(path: Path) -> dict:
    from tttrkit.ptuio.decoder import get_photons, resolve_markers

    reader, corr = _open(path)
    line_nsyncs, frame_nsyncs = [], []
    max_channel = 0
    for chunk in reader.iter_chunks(CHUNK_SIZE):
        ev = corr.correct(chunk)
        frame_m, start_m, _ = resolve_markers(ev, 4, 1, 2)
        line_nsyncs.append(start_m["nsync"])
        frame_nsyncs.append(frame_m["nsync"])
        photons = get_photons(ev)
        if len(photons):
            max_channel = max(max_channel, int(photons["channel"].max()))
    line_nsyncs = np.concatenate(line_nsyncs)
    n_lines = len(line_nsyncs)
    if n_lines < 2:
        raise SkipCase(f"only {n_lines} line start marker(s)")

    tags = reader.header.tags
    if tags.get("ImgHdr_PixY"):
        # Geometry recorded by the acquisition software (e.g. Abberior)
        lines = int(tags["ImgHdr_PixY"])
    else:
        # Line count between distinct frame-marker positions; segments much
        # shorter than the longest (start glitches, duplicates) are ignored
        pos = np.searchsorted(line_nsyncs, np.concatenate(frame_nsyncs), side="left")
        bounds = np.unique(np.concatenate([[0, n_lines], pos]))
        seg = np.diff(bounds)
        lines = int(np.median(seg[seg >= 0.1 * seg.max()]))
    frames = max(1, int(np.ceil(n_lines / lines - 0.1)))
    pix_x = tags.get("ImgHdr_PixX")
    return {
        "lines": lines,
        "pixels": int(pix_x) if pix_x else 256,
        "frames": frames,
        "max_detector": min(max_channel + 1, MAX_DETECTOR_CAP),
    }


def _hash(a: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}{a.shape}".encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def prescan(reader, cfg, chunk_size: int):
    """First-frame pre-scan (fills cfg.line_duration); None on an older
    tttrkit, whose ImageReconstructor estimated the line duration itself."""
    try:
        from tttrkit.ptuio.prescan import prescan_first_frame
    except ImportError:
        return None
    return prescan_first_frame(reader, cfg, chunk_size=chunk_size)


def reconstruct(path: Path, config: dict, recon_kwargs: dict, chunk_size: int) -> dict:
    from tttrkit.ptuio.reconstructor import ImageReconstructor, ScanConfig

    rk = {"outputs": ["photon_count", "mean_arrival_time", "tcspc_histogram"]}
    rk.update(recon_kwargs)
    rk.setdefault(
        "tcspc_bin_factor", max(1, rk.get("tcspc_channels", 2**15) // 512)
    )
    cfg = ScanConfig(**_supported_kwargs(ScanConfig.__init__, config))

    reader, corr = _open(path)
    out = io.StringIO()
    with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(out):
        warnings.simplefilter("always")
        scan = prescan(reader, cfg, chunk_size)
        rec = ImageReconstructor(
            cfg, **_supported_kwargs(ImageReconstructor.__init__, rk)
        )
        for chunk in reader.iter_chunks(chunk_size):
            rec.update(corr.correct(chunk))
        ds = rec.finalize()

    pc = ds["photon_count"].values  # (frame, sequence, line, pixel, channel)
    line_profile = pc.sum(axis=(3, 4), dtype=np.int64)
    events = getattr(rec, "frame_sync_events", [])
    printed = sorted(
        {l.strip() for l in out.getvalue().splitlines()}
        - {"", "Reconstruction finished."}
    )
    return {
        "hashes": {
            name: _hash(ds[name].values)
            for name in sorted(ds.data_vars)
            if ds[name].ndim > 0
        },
        "line_profile": line_profile,
        "column_profile": pc.sum(axis=(0, 2), dtype=np.int64),
        "tcspc_profile": ds["tcspc_histogram"].values.sum(axis=0),
        "total_photons": int(pc.sum(dtype=np.int64)),
        "line_duration": getattr(rec, "line_duration", None)
        or getattr(cfg, "line_duration", None),
        "empty_lines": int((line_profile == 0).sum()),
        "frame_sync": {
            a: sum(e["action"] == a for e in events)
            for a in sorted({e["action"] for e in events})
        },
        # informational only, not compared
        "prescan": None if scan is None else {
            "verdict": scan.verdict,
            "measured_lines_per_frame": scan.measured_lines_per_frame,
            "line_duration_source": scan.line_duration_source,
        },
        "data_quality": {k: int(v) for k, v in ds.attrs.items()},
        "frame_sync_mismatches": [int(e["mismatch"]) for e in events if e["mismatch"]],
        "warnings": [f"{w.category.__name__}: {w.message}"[:300] for w in caught],
        "printed": [p[:300] for p in printed],
    }


def run_case(case: Case, data_dir: Path) -> dict:
    """Reconstruct one case. Never raises; failures go into the result."""
    t0 = time.time()
    result = {"id": case.id, "status": "ok", "config_source": case.source}
    path = Path(data_dir) / case.path
    try:
        if not path.exists():
            raise SkipCase(f"file not found: {path}")
        config = case.config if case.config is not None else infer_config(path)
        config = {**config, **case.overrides}
        result["config"] = config
        result["reconstructor"] = case.reconstructor
        first = reconstruct(path, config, case.reconstructor, case.chunk_sizes[0])
        result.update(first)
        result["chunk_mismatch"] = [
            cs
            for cs in case.chunk_sizes[1:]
            if reconstruct(path, config, case.reconstructor, cs)["hashes"]
            != first["hashes"]
        ]
    except SkipCase as e:
        result.update(status="skip", reason=str(e))
    except Exception:
        result.update(status="error", error=traceback.format_exc(limit=-4))
    result["elapsed"] = round(time.time() - t0, 2)
    return result


def _run_case_args(args):
    return run_case(*args)


def run_cases(cases: list[Case], data_dir: Path, workers: int = DEFAULT_WORKERS):
    """Run cases in parallel, yielding results as they complete."""
    jobs = [(c, Path(data_dir)) for c in cases]
    if workers <= 1:
        yield from map(_run_case_args, jobs)
        return
    # Largest files first, for better load balancing
    jobs.sort(key=lambda j: -_size(j[1] / j[0].path))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(_run_case_args, jobs)


def _size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0


# ----------------------------------------------------------------------
# References
# ----------------------------------------------------------------------
ARRAYS = ("line_profile", "column_profile", "tcspc_profile")


def _git_describe() -> str:
    """Version of the tttrkit code under test (not of this test file)."""
    try:
        import subprocess

        import tttrkit

        return subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=Path(tttrkit.__file__).parent,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def save_reference(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in result.items() if k not in ARRAYS}
    meta["config_source"] = (
        "inferred" if result["config_source"] in ("inferred", "reference")
        else result["config_source"]
    )
    meta["created_with"] = _git_describe()
    np.savez_compressed(
        path,
        meta=np.array(json.dumps(meta, default=str)),
        **{k: result[k] for k in ARRAYS},
    )


def load_reference(path: Path) -> dict:
    with np.load(path) as z:
        ref = {k: z[k] for k in ARRAYS if k in z}
        ref["meta"] = json.loads(str(z["meta"]))
    return ref


def compare(ref: dict, new: dict) -> list[str]:
    """Differences between a stored reference and a new result ([] = same)."""
    meta = ref["meta"]
    diffs = []
    if meta.get("config") != new.get("config") or meta.get(
        "reconstructor"
    ) != new.get("reconstructor"):
        diffs.append(
            f"CONFIG CHANGED: reference {meta.get('config')} "
            f"{meta.get('reconstructor')} -> now {new.get('config')} "
            f"{new.get('reconstructor')}"
        )
        return diffs

    if meta["hashes"] != new["hashes"]:
        changed = [k for k in new["hashes"] if meta["hashes"].get(k) != new["hashes"][k]]
        d_total = new["total_photons"] - meta["total_photons"]
        head = f"OUTPUT CHANGED ({', '.join(changed)}): photons {d_total:+d}"
        a, b = ref["line_profile"], new["line_profile"]
        if a.shape != b.shape:
            head += f"; line profile shape {a.shape} -> {b.shape}"
        else:
            d = b.astype(np.int64) - a
            idx = np.argwhere(d)
            head += f", {len(idx)} of {d.size} (frame, sequence, line) changed"
            if len(idx):
                head += f", first {[tuple(map(int, i)) for i in idx[:5]]}"
        diffs.append(head)
        if new["empty_lines"] != meta["empty_lines"]:
            diffs.append(f"empty lines {meta['empty_lines']} -> {new['empty_lines']}")
        if new.get("line_duration") != meta.get("line_duration"):
            diffs.append(
                f"line duration {meta.get('line_duration')} -> {new.get('line_duration')}"
            )

    if meta.get("frame_sync", {}) != new.get("frame_sync", {}):
        diffs.append(
            f"FRAME SYNC CHANGED: {meta.get('frame_sync')} -> {new.get('frame_sync')}"
        )
    if diffs:
        for key in ("warnings", "printed"):
            old, cur = set(meta.get(key, [])), set(new.get(key, []))
            diffs += [f"  - {key}: {m}" for m in sorted(old - cur)]
            diffs += [f"  + {key}: {m}" for m in sorted(cur - old)]
        diffs.append(f"  reference created with {meta.get('created_with')}")
    return diffs


# ----------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("data_dir", type=Path, nargs="?", default=SAMPLE_DIR)
    p.add_argument("--refs", type=Path, default=None)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--update", action="store_true", help="rewrite all references")
    g.add_argument("--add-missing", action="store_true", help="create missing references")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--filter", default="", help="only case ids containing this")
    p.add_argument("--jsonl", type=Path, help="also write every result (no arrays) here")
    args = p.parse_args(argv)
    jsonl = open(args.jsonl, "w", encoding="utf-8") if args.jsonl else None

    refs = args.refs or default_refs(args.data_dir)
    cases = [
        c for c in discover_cases(args.data_dir, refs)
        if args.filter.lower() in c.id.lower()
    ]
    by_id = {c.id: c for c in cases}
    counts = {}
    for res in run_cases(cases, args.data_dir, args.workers):
        case = by_id[res["id"]]
        ref_path = case.reference_path(refs)
        if res["status"] != "ok":
            outcome = res["status"].upper()
            detail = res.get("reason") or res.get("error", "").strip().splitlines()[-1]
        elif args.update or (args.add_missing and not ref_path.exists()):
            save_reference(ref_path, res)
            outcome, detail = "SAVED", ""
        elif not ref_path.exists():
            outcome, detail = "MISSING", "no reference"
        else:
            diffs = compare(load_reference(ref_path), res)
            outcome = "CHANGED" if diffs else "same"
            detail = "\n      ".join(diffs)
        counts[outcome] = counts.get(outcome, 0) + 1
        if outcome != "same":
            print(f"{outcome:8s} {res['id']} ({res['elapsed']}s) {detail}", flush=True)
        if jsonl:
            row = {k: v for k, v in res.items() if k not in ARRAYS}
            jsonl.write(json.dumps(dict(row, outcome=outcome), default=str) + "\n")
    if jsonl:
        jsonl.close()
    print("summary:", counts)
    return 0 if set(counts) <= {"same", "SAVED", "SKIP"} else 1


if __name__ == "__main__":
    sys.exit(main())
