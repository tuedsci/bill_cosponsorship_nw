"""
Module for generic helper functions.

Author: Tue Nguyen
"""

import io
import zipfile
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests
from tqdm import tqdm


# --------------------------------------------------
# GENERIC FUNCTIONS
# --------------------------------------------------
def load_data(data_path: str, **kwargs) -> pd.DataFrame:
    """
    Load data from multiple file formats.
    """
    data_path = Path(data_path)
    file_ext = data_path.suffix

    if file_ext == ".csv":
        df = pd.read_csv(data_path, **kwargs)
    elif file_ext == ".parquet":
        df = pd.read_parquet(data_path, **kwargs)
    elif file_ext == ".feather":
        df = pd.read_feather(data_path, **kwargs)
    elif file_ext in [".xls", ".xlsx"]:
        df = pd.read_excel(data_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

    # Replace None with proper pd.NA
    df = df.where(pd.notnull(df), pd.NA)

    return df


# --------------------------------------------------
# DOWNLOAD DATA
# --------------------------------------------------
def download_file(
    url: str,
    save_path: str | Path,
    overwrite: bool = False,
    chunk_mb: int = 1,
) -> bool:
    """
    Download and save a file from a URL.
    Args:
        url: URL to download from
        save_path: Path to save the file
        overwrite: If True, overwrite existing file
        chunk_mb: Size of download chunks in MB
    """
    if isinstance(save_path, str):
        save_path = Path(save_path)

    if save_path.exists() and not overwrite:
        print(f"{save_path.name}: File exists. Skip.")
        return False

    chunk_size = chunk_mb * 1024 * 1024

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }

    with requests.Session() as session:
        response = session.get(url, stream=True, headers=headers)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f, tqdm(
            desc=f"Downloading {save_path.name}",
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
            leave=False,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                size = f.write(chunk)
                progress_bar.update(size)

    return True


# --------------------------------------------------
# ZIP FILE
# --------------------------------------------------
def apply_to_zip(
    zip_path: str | Path,
    func: Callable,
    file_pattern: str = "*",
    **kwargs,
) -> list[Any]:
    """
    Apply a function to all files in a zip file matching a pattern.
    Args:
        zip_path: Path to the zip file
        func: Function to apply to each file
        file_pattern: Pattern to match files (e.g., "*.xml", "*.txt")
        **kwargs: Additional kwargs
    """
    results = []

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        # Filter files matching pattern
        pattern = Path(file_pattern).name  # Get just the file pattern part
        matching_files = [
            f for f in zip_ref.namelist() if f.endswith(pattern.replace("*", ""))
        ]

        # Process each matching file
        for file_name in tqdm(matching_files, leave=False):
            try:
                with zip_ref.open(file_name) as file:
                    temp_file = io.BytesIO(file.read())
                    temp_file.name = Path(file_name).name
                    result = func(temp_file, **kwargs)
                    results.append(result)
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
                continue

    return results
