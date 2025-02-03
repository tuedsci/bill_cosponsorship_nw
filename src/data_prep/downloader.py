"""
Module to download data.

Author: Tue Nguyen
"""

from pathlib import Path

from src.config import cfg
from src.utils import helper


# --------------------------------------------------
# DOWNLOAD BILL STATUS DATA
# --------------------------------------------------
def download_bill_status_data(
    congresses: list[int],
    bill_types: list[str] = cfg.VALID_BILL_TYPES,
    overwrite: bool = False,
    save_dir: Path = cfg.dir.DATA_RAW / "bill_status",
) -> None:
    """
    Download bill status data from the www.govinfo.gov.

    Args:
        congresses: List of congresses to download.
        bill_types: List of bill types to download.
        overwrite: If True, overwrite existing files.
        data_dir: Directory to save the downloaded data
    """
    # Validate bill types
    for bill_type in bill_types:
        if bill_type not in cfg.VALID_BILL_TYPES:
            raise ValueError(f"Invalid bill type: {bill_type}")

    # Create directory for downloaded data
    save_dir.mkdir(parents=True, exist_ok=True)

    # Download data
    for congress in congresses:
        congress = str(congress).lower()
        for bill_type in bill_types:
            bill_type = bill_type.lower()
            file_name = f"{congress}_{bill_type}.zip"
            save_path = save_dir / file_name
            url = (
                f"{cfg.url.US_BILL_STATUS}/{congress}/{bill_type}/"
                f"BILLSTATUS-{congress}-{bill_type}.zip"
            )

            helper.download_file(url, save_path, overwrite=overwrite)

    print(f"Download completed. Check in {save_dir}")
