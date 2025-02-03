from pathlib import Path
from types import SimpleNamespace

cfg = SimpleNamespace(
    # DIRECTORIES
    dir=SimpleNamespace(
        DATA_ROOT=Path("data"),
        DATA_RAW=Path("data") / "raw",
        DATA_CLEAN=Path("data") / "clean",
        DATA_TEMP=Path("data") / "temp",
    ),
    # URLS
    url=SimpleNamespace(
        US_BILL_STATUS="https://www.govinfo.gov/bulkdata/BILLSTATUS",
    ),
    # COLOR MAPS
    cmap=SimpleNamespace(
        PARTY={
            # Blue
            "D": "#0044CC",
            "ID": "#6699CC",
            # Red
            "R": "#DE0100",
            # Green
            "I": "#2E8B57",
            # Gold
            "L": "#E6B700",
        },
    ),
    # OTHER
    VALID_BILL_TYPES=(
        "hr",
        "s",
        "hjres",
        "sjres",
        "hconres",
        "sconres",
        "hres",
        "sres",
    ),
)
