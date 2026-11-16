"""Regression tests of ImageReconstructor on raw .ptu data.

    pytest tests/test_realdata.py                    # sample sets (clean + anomalies)
    pytest ... -k clean                              # one sample set
    pytest ... --realdata C:/path/to/Data            # every file in a folder
    pytest ... --update-references       # accept intended behaviour changes
    pytest ... --add-missing-references  # fingerprint newly added files

The sample sets live in tests/realdata_references (git-ignored, local to
this machine); without them and without --realdata the tests are skipped.
See realdata_harness.py for how cases and configs are resolved.
"""

from pathlib import Path

import pytest

import realdata_harness as h


def _data_dir(config):
    data = config.getoption("--realdata")
    if data:
        return Path(data)
    if h.find_ptus(h.SAMPLE_DIR) if h.SAMPLE_DIR.exists() else []:
        return h.SAMPLE_DIR
    return None


def _opts(config):
    data = _data_dir(config)
    refs = config.getoption("--realdata-refs")
    return {
        "data": data,
        "refs": Path(refs) if refs else (h.default_refs(data) if data else None),
        "workers": config.getoption("--realdata-workers") or h.DEFAULT_WORKERS,
        "update": config.getoption("--update-references"),
        "add_missing": config.getoption("--add-missing-references"),
    }


def pytest_generate_tests(metafunc):
    if "case" not in metafunc.fixturenames:
        return
    opts = _opts(metafunc.config)
    if not opts["data"]:
        metafunc.parametrize(
            "case",
            [pytest.param(None, marks=pytest.mark.skip(
                reason="no sample sets in tests/realdata_references; pass "
                "--realdata DIR or set TTTRKIT_DATA_DIR"
            ))],
            ids=["no-data"],
        )
        return
    cases = h.discover_cases(opts["data"], opts["refs"])
    metafunc.parametrize("case", cases, ids=[c.id for c in cases])


@pytest.fixture(scope="session")
def realdata_results(request):
    """Reconstruct every selected case once, in parallel."""
    opts = _opts(request.config)
    selected = [
        item.callspec.params["case"]
        for item in request.session.items
        if hasattr(item, "callspec") and item.callspec.params.get("case") is not None
    ]
    return {
        r["id"]: r
        for r in h.run_cases(selected, Path(opts["data"]), opts["workers"])
    }


@pytest.mark.realdata
def test_reconstruction(case, realdata_results, request):
    opts = _opts(request.config)
    res = realdata_results[case.id]
    if res["status"] == "skip":
        pytest.skip(res["reason"])
    if res["status"] == "error":
        pytest.fail(f"reconstruction raised:\n{res['error']}", pytrace=False)

    problems = []
    if res["chunk_mismatch"]:
        problems.append(
            f"output depends on chunk size: {case.chunk_sizes[0]} vs "
            f"{res['chunk_mismatch']}"
        )
    exp = case.expect
    n_sync_warn = sum(w.startswith("FrameSyncWarning") for w in res["warnings"])
    if "frame_sync_warnings" in exp and n_sync_warn != exp["frame_sync_warnings"]:
        problems.append(
            f"expected {exp['frame_sync_warnings']} FrameSyncWarning(s), got "
            f"{n_sync_warn}: {res['warnings']}"
        )
    if "empty_lines" in exp and res["empty_lines"] != exp["empty_lines"]:
        problems.append(f"expected {exp['empty_lines']} empty lines, got {res['empty_lines']}")
    if "frame_sync" in exp and res["frame_sync"] != exp["frame_sync"]:
        problems.append(f"expected frame sync {exp['frame_sync']}, got {res['frame_sync']}")

    ref_path = case.reference_path(opts["refs"])
    if opts["update"] or (opts["add_missing"] and not ref_path.exists()):
        h.save_reference(ref_path, res)
    elif not ref_path.exists():
        problems.append(
            "no reference fingerprint; review the output, then run with "
            "--add-missing-references"
        )
    else:
        problems += h.compare(h.load_reference(ref_path), res)

    if problems:
        pytest.fail("\n".join(problems), pytrace=False)
