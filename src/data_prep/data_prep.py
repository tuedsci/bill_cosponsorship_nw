"""
Module to clean and preparing data.

Author: Tue Nguyen
"""

import io
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config import cfg
from src.utils import helper


# --------------------------------------------------
# XML PARSING HELPERS
# --------------------------------------------------
def _get_bill_info(bill: ET.Element) -> dict:
    """
    Extract key information from a bill XML element.
    Args:
        bill: XML element containing bill information.
    """
    info = {}
    info["congress"] = bill.findtext("congress")
    info["bill_type"] = bill.findtext("type") or bill.findtext("billType")
    info["bill_number"] = bill.findtext("number") or bill.findtext("billNumber")
    info["origin_chamber"] = bill.findtext("originChamber")
    info["policy_area"] = bill.findtext("policyArea/name")
    info["title"] = bill.findtext("title")
    info["introduced_date"] = bill.findtext("introducedDate")

    return info


def _get_sponsor_info(
    sponsor: ET.Element,
    sponsor_type: str,
) -> dict:
    """
    Extract sponsor information from an XML element.
    Args:
        sponsor: XML element containing sponsor information.
        sponsor_type: Type of sponsor ('sponsor' or 'cosponsor').
    """
    info = {}
    info["bioguide_id"] = sponsor.findtext("bioguideId")
    info["full_name"] = sponsor.findtext("fullName")
    info["first_name"] = sponsor.findtext("firstName")
    info["last_name"] = sponsor.findtext("lastName")
    info["party"] = sponsor.findtext("party")
    info["state"] = sponsor.findtext("state")
    info["district"] = sponsor.findtext("district")
    info["sponsor_type"] = sponsor_type

    return info


# --------------------------------------------------
# POST-PROCESSING HELPERS
# --------------------------------------------------
def _create_bill_id(
    congress: pd.Series,
    bill_type: pd.Series,
    bill_number: pd.Series,
) -> pd.Series:
    """
    Create a bill ID from congress, bill type, and bill number.  Example: "115_hr_1"
    Args:
        congress: Series of congress numbers
        bill_type: Series of bill types
        bill_number: Series of bill numbers
    """
    return (congress + "_" + bill_type + "_" + bill_number).str.lower()


def _get_sponsor_title(name: pd.Series) -> pd.Series:
    """
    Extract sponsor title from full name. Example: "Sen. John Doe" -> "Sen"
    """
    title = name.str.split(".").str[0].str.strip().str.title().astype("string")
    title[title.str.startswith("Res")] = "Res"
    return title


def _create_sponsor_id(
    bioguide_id: pd.Series,
    party: pd.Series,
    state: pd.Series,
    title: pd.Series,
) -> pd.Series:
    """
    Create sponsor ID from display name and bioguide ID.
    Example: "Sen. John Doe (D), A000360"
    Args:
        bioguide_id: Series of bioguide IDs
        party: Series of party affiliations
        title: Series of sponsor titles
    """
    return bioguide_id + "_" + party + "_" + state + "_" + title


def _post_process_bills(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-process bill data.
    """
    # Convert to proper data types and formats
    for col in ["bill_type", "origin_chamber", "policy_area"]:
        df[col] = df[col].str.lower()

    for col in ["congress", "bill_number"]:
        df[col] = df[col].astype("Int64")

    df["introduced_date"] = pd.to_datetime(df["introduced_date"])

    # Create bill ID
    df["bill_id"] = _create_bill_id(
        df["congress"].astype("string"),
        df["bill_type"],
        df["bill_number"].astype("string"),
    )

    return df


def _post_process_sponsorship(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-process sponsorship data.
    """
    # Extract sponsor title
    df["sponsor_title"] = _get_sponsor_title(df["full_name"])
    del df["full_name"]

    # Convert to proper data types and formats
    for col in ["party", "state", "district"]:
        df[col] = df[col].str.upper()

    for col in ["first_name", "last_name"]:
        df[col] = df[col].str.title()

    df["sponsor_id"] = _create_sponsor_id(
        df["bioguide_id"],
        df["party"],
        df["state"],
        df["sponsor_title"],
    )

    return df


# --------------------------------------------------
# DATA EXTRACTION
# --------------------------------------------------
def extract_bill_info(xml_path: str | Path | io.BytesIO) -> dict:
    """
    Extract key information from a bill XML file.
    """
    # Load XML file
    if isinstance(xml_path, str):
        xml_path = Path(xml_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Basic bill info
    bill = _get_bill_info(root.find("bill"))

    # Sponsors and cosponsors
    sponsors = [
        _get_sponsor_info(s, sponsor_type="sponsor")
        for s in root.findall("bill/sponsors/item")
    ]

    sponsors += [
        _get_sponsor_info(s, sponsor_type="cosponsor")
        for s in root.findall("bill/cosponsors/item")
    ]

    # Attach sponsors to bill info
    bill["sponsors"] = sponsors

    return bill


def process_all_bills(
    data_dir: str | Path = cfg.dir.DATA_RAW / "bill_status",
    save_dir: str | Path = cfg.dir.DATA_CLEAN,
    dry_run: bool = False,
) -> None | pd.DataFrame:
    """
    Process all bill XML files from zip files in the given directory.
    Args:
        data_dir: Directory containing zip files with bill XML files.
        save_dir: Directory to save the processed data.
        dry_run: Run for subset of to speed up testing (no saving).
    """
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------
    # COLLECT DATA FROM ZIP FILES
    # ----------------------------------------------
    bills = []
    zip_files = list(data_dir.glob("*.zip"))

    # For testing, use first two zip files to speed up
    if dry_run:
        zip_files = zip_files[:2]

    # Extract bill info from all zip files
    for zip_file in tqdm(zip_files):
        data = helper.apply_to_zip(zip_file, extract_bill_info)
        bills.extend(data)

    # ----------------------------------------------
    # POST-PROCESS AND SAVE
    # ----------------------------------------------
    print("Post-processing data...")
    # Convert to dataframe
    meta_cols = [col for col in bills[0].keys() if col != "sponsors"]
    bills = pd.json_normalize(bills, record_path="sponsors", meta=meta_cols)

    # Make sure all columns are proper strings
    for col in bills.columns:
        bills[col] = bills[col].astype("string").str.strip()

    # Post-process data
    bills = _post_process_bills(bills)
    bills = _post_process_sponsorship(bills)
    bills.drop_duplicates(subset=["bill_id", "sponsor_id"], inplace=True)

    if dry_run:
        return bills

    bills.to_parquet(save_dir / "bills_sponsors.parquet")
    print(f"Done. Check in {save_dir}.")


# --------------------------------------------------
# PREPARE FINAL DATA
# --------------------------------------------------
def apply_main_filters(
    df: pd.DataFrame,
    congresses: list[int],
    bill_types: list[str],
    policy_areas: list[str],
) -> pd.DataFrame:
    """
    Apply the main filters on congresses, bill types, and policy areas.
    Args:
        congresses: List of congresses to include
        bill_types: List of bill types to include
        policy_areas: List of policy areas to include
    """
    df = df.copy()

    if congresses:
        df = df[df["congress"].isin(congresses)]
    if bill_types:
        df = df[df["bill_type"].isin(bill_types)]
    if policy_areas:
        df = df[df["policy_area"].isin(policy_areas)]

    return df


def prepare_data(
    data_path: str | Path = cfg.dir.DATA_CLEAN / "bills_sponsors.parquet",
    congresses: list[int] | None = None,
    bill_types: list[str] | None = None,
    policy_areas: list[str] | None = None,
) -> pd.DataFrame:
    """
    Prepare data pipeline for analysis.
    Args:
        data_path: Path to the data file
        congresses: List of congresses to include
        bill_types: List of bill types to include
        policy_areas: List of policy areas to include
    """
    # Load and filter data
    df = pd.read_parquet(data_path)
    df = apply_main_filters(df, congresses, bill_types, policy_areas)

    # --------------------------------------------------
    # AHOC FIXES
    # --------------------------------------------------
    # Fix inconsistency in sponsor names
    name_df = df[["bioguide_id", "first_name", "last_name"]].drop_duplicates(
        subset=["bioguide_id"]
    )

    df = df.drop(columns=["first_name", "last_name"]).merge(name_df, on="bioguide_id")

    # Fix sponsor_title to "Rep" for every House bill with "Sen" title
    # Then regenerate sponsor_id because it's based on sponsor_title
    cond = (df["origin_chamber"] == "house") & (df["sponsor_title"] == "Sen")
    df.loc[cond, "sponsor_title"] = "Rep"

    df["sponsor_id"] = _create_sponsor_id(
        df["bioguide_id"],
        df["party"],
        df["state"],
        df["sponsor_title"],
    )

    # --------------------------------------------------
    # ADD EXTRA COLUMNS
    # --------------------------------------------------
    df["display_name"] = (
        df["sponsor_title"]
        + ". "
        + df["first_name"]
        + " "
        + df["last_name"]
        + " ("
        + df["party"]
        + ")"
    )

    return df
