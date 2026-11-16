import os


def pytest_addoption(parser):
    g = parser.getgroup("realdata", "real-data regression tests")
    g.addoption(
        "--realdata",
        default=os.environ.get("TTTRKIT_DATA_DIR"),
        metavar="DIR",
        help="directory searched for .ptu files (default: $TTTRKIT_DATA_DIR, "
        "else the sample sets in tests/realdata_references if present)",
    )
    g.addoption(
        "--realdata-refs",
        default=os.environ.get("TTTRKIT_REFERENCE_DIR"),
        metavar="DIR",
        help="reference fingerprints (default: tests/realdata_references/"
        "fingerprints for the sample sets, .../corpus_fingerprints otherwise)",
    )
    g.addoption(
        "--realdata-workers",
        type=int,
        default=None,
        metavar="N",
        help="parallel reconstruction processes",
    )
    g.addoption(
        "--update-references",
        action="store_true",
        help="accept current behaviour: rewrite references of the selected cases",
    )
    g.addoption(
        "--add-missing-references",
        action="store_true",
        help="create references only for cases that have none",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "realdata: regression test on raw .ptu data"
    )
