"""Data parsing, feature extraction, and target labeling routines.
Provides robust handling of heterogeneous store metadata.
"""

import re
from typing import Tuple
import numpy as np
import pandas as pd


def parse_size_in_mb(val) -> float:
    """Robust parser for app client sizes across formats (bytes, KB, MB, GB strings).
    Returns size in megabytes as a float.
    """
    if pd.isna(val):
        return np.nan

    # If already a number
    if isinstance(val, (int, float)):
        # If value is large (e.g., raw bytes > 10000)
        if val > 10000:
            return float(val) / (1024.0 * 1024.0)
        return float(val)

    s = str(val).strip().lower()
    if s in ["varies with device", "nan", "none", ""]:
        return np.nan

    # Strip commas
    s = s.replace(",", "")

    # Match numeric portion and optional unit
    m = re.match(r"^([\d\.]+)\s*([a-z]*)$", s)
    if not m:
        return np.nan

    number_part, unit_part = m.groups()
    try:
        num = float(number_part)
    except ValueError:
        return np.nan

    unit = unit_part.lower()
    if unit in ["k", "kb"]:
        return num / 1024.0
    elif unit in ["m", "mb", ""]:
        # If no unit and small, assume MB; if huge (>10000), assume bytes
        if unit == "" and num > 10000:
            return num / (1024.0 * 1024.0)
        return num
    elif unit in ["g", "gb"]:
        return num * 1024.0
    return num


def parse_max_iap(val) -> float:
    """Parse comma-separated in-app purchase price tiers and return the maximum price."""
    if pd.isna(val):
        return 0.0
    try:
        prices = [float(x.strip()) for x in str(val).split(",") if x.strip()]
        return max(prices) if prices else 0.0
    except (ValueError, TypeError):
        return 0.0


def extract_genres(genres_val) -> Tuple[str, str]:
    """Extract primary and secondary game genres from comma-separated genre strings."""
    if pd.isna(genres_val):
        return "Strategy", "None"
    parts = [p.strip() for p in str(genres_val).split(",") if p.strip().lower() != "games"]
    primary = parts[0] if len(parts) > 0 else "Strategy"
    secondary = parts[1] if len(parts) > 1 else "None"
    return primary, secondary


def estimate_minimum_os(row: pd.Series) -> str:
    """Estimate minimum supported OS version from description requirements or release epoch."""
    desc = str(row.get("Description", ""))
    match = re.search(r"requires\s+(?:ios|iphone os)\s*([\d\.]+)", desc, re.I)
    if match:
        major = match.group(1).split(".")[0]
        return f"iOS {major}.0"

    orig_date = row.get("orig_date", pd.NaT)
    year = orig_date.year if pd.notna(orig_date) else 2016
    if year <= 2011:
        return "iOS 5.0"
    elif year <= 2013:
        return "iOS 7.0"
    elif year <= 2015:
        return "iOS 9.0"
    elif year <= 2017:
        return "iOS 10.0"
    else:
        return "iOS 11.0"


def determine_session_pacing(description: str, primary_genre: str) -> str:
    """Classify game pacing architecture based on genre and loop mechanics."""
    t = str(description).lower()
    if any(k in t for k in ["idle", "clicker", "hypercasual", "endless runner", "tap to play"]):
        return "Hypercasual/Idle"
    elif any(k in t for k in ["turn-based", "turn based", "chess", "puzzle", "card game", "board"]):
        return "Turn-Based"
    elif any(k in t for k in ["real-time", "real time", "action", "shooter", "racing", "arcade"]):
        return "Real-Time"
    else:
        return "Session-Based"


def parse_languages_count(val) -> int:
    """Return count of supported localization languages."""
    if pd.isna(val):
        return 1
    langs = [x.strip() for x in str(val).split(",") if x.strip()]
    return max(1, len(langs))


def process_raw_dataset(df_raw: pd.DataFrame, reference_date: str = "2019-09-01") -> pd.DataFrame:
    """Transform raw store metadata into standardized, clean feature matrix with binary labels.
    Strictly filters out ambiguous boundary entries.
    """
    df = df_raw.copy()

    # Parse financial ground truths
    df["max_iap_price"] = df["In-app Purchases"].apply(parse_max_iap)
    df["has_iap"] = df["In-app Purchases"].notna() & (df["max_iap_price"] > 0)
    df["is_free"] = (df["Price"] == 0) | (df["Price"].isna())

    # Detect advertisement monetization signals
    ad_pattern = re.compile(
        r"\b(?:ads?|advertis(?:ing|ement|ements)|ad-supported|ad-free|remove ads?|no ads?|interstitial|banner ads?|rewarded video)\b",
        re.I,
    )
    df["desc_mentions_ads"] = df["Description"].str.contains(ad_pattern, na=False)
    df["contains_ads"] = df["desc_mentions_ads"] | (df["is_free"] & (df["max_iap_price"] < 4.99))

    # Formulate binary target definitions
    is_aggressive_iap = df["has_iap"] & (df["max_iap_price"] >= 19.99)
    is_ad_supported = df["contains_ads"] & (df["max_iap_price"] < 4.99)

    # Filter to unambiguous boundary classes
    retained_mask = is_aggressive_iap | is_ad_supported
    df_clean = df[retained_mask].copy()
    df_clean["target"] = np.where(df_clean["has_iap"] & (df_clean["max_iap_price"] >= 19.99), 1, 0)

    # Feature 1: size_in_mb
    df_clean["size_in_mb"] = df_clean["Size"].apply(parse_size_in_mb).round(2)
    median_size = df_clean["size_in_mb"].median()
    df_clean["size_in_mb"] = df_clean["size_in_mb"].fillna(median_size)

    # Feature 2 & 3: primary and secondary genre
    genre_pairs = df_clean["Genres"].apply(extract_genres)
    df_clean["primary_genre"] = [p[0] for p in genre_pairs]
    df_clean["secondary_genre"] = [p[1] for p in genre_pairs]

    # Feature 4: content rating
    df_clean["content_rating"] = df_clean["Age Rating"].fillna("4+").astype(str)

    # Feature 5: minimum OS version
    df_clean["orig_date"] = pd.to_datetime(
        df_clean["Original Release Date"], format="%d/%m/%Y", errors="coerce"
    )
    df_clean["curr_date"] = pd.to_datetime(
        df_clean["Current Version Release Date"], format="%d/%m/%Y", errors="coerce"
    )
    df_clean["minimum_os_version"] = df_clean.apply(estimate_minimum_os, axis=1)

    # Feature 6: synchronous multiplayer
    sync_pattern = re.compile(
        r"\b(?:real-time|real time|synchronous|pvp|multiplayer|online battle|co-op|guild|clan|mmo|matchmaking|arena)\b",
        re.I,
    )
    df_clean["is_synchronous_multiplayer"] = df_clean["Description"].str.contains(
        sync_pattern, na=False
    ).astype(int)

    # Feature 7: session pacing tag
    df_clean["session_pacing_tag"] = df_clean.apply(
        lambda r: determine_session_pacing(r["Description"], r["primary_genre"]), axis=1
    )

    # Feature 8: supported languages count
    df_clean["supported_languages_count"] = df_clean["Languages"].apply(parse_languages_count)

    # Feature 9: days since last update
    ref_dt = pd.to_datetime(reference_date)
    df_clean["days_since_last_update"] = (
        (ref_dt - df_clean["curr_date"]).dt.days.clip(lower=0).fillna(365).astype(int)
    )

    keep_cols = [
        "primary_genre",
        "secondary_genre",
        "minimum_os_version",
        "content_rating",
        "size_in_mb",
        "is_synchronous_multiplayer",
        "session_pacing_tag",
        "supported_languages_count",
        "days_since_last_update",
        "target",
    ]

    return df_clean[keep_cols].reset_index(drop=True)
