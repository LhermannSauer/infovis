import pandas as pd
import requests
import time
import re
from typing import List, Dict
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed


load_dotenv()


# -----------------------------
# CONFIG
# -----------------------------
INPUT_CSV = "../data/bronze/spotify_history.csv"
OUTPUT_PARQUET = "../data/silver/1.enriched_tracks.parquet"

# Optional: Last.fm API (get one free)
LASTFM_API_KEY = os.getenv("LASTFM_KEY")
MAX_WORKERS = 15

# -----------------------------
# LOAD DATA
# -----------------------------

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# -----------------------------
# FILTER TRACKS
# -----------------------------

def filter_tracks(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["spotify_track_uri"].notna()]
    df = df[df["master_metadata_track_name"].notna()]
    return df

# -----------------------------
# EXTRACT TRACK ID
# -----------------------------

def extract_track_id(uri: str) -> str:
    if pd.isna(uri):
        return None
    match = re.search(r"track:([a-zA-Z0-9]+)", uri)
    return match.group(1) if match else None

# -----------------------------
# PREPROCESS (DEDUPLICATE)
# -----------------------------

def preprocess(df: pd.DataFrame):
    df["track_id"] = df["spotify_track_uri"].apply(extract_track_id)
    df = df[df["track_id"].notna()]

    # Aggregate original dataset (keep behavior data)
    agg = df.groupby([
        "track_id",
        "master_metadata_track_name",
        "master_metadata_album_artist_name"
    ]).agg({
        "ms_played": "sum",
        "ts": "count"
    }).reset_index()

    agg = agg.rename(columns={
        "ts": "play_count",
        "ms_played": "total_ms"
    })

    # Deduplicated set for API calls
    unique_tracks = agg[[
        "track_id",
        "master_metadata_track_name",
        "master_metadata_album_artist_name"
    ]].drop_duplicates().reset_index(drop=True)

    return agg, unique_tracks

# -----------------------------
# LAST.FM REQUEST
# -----------------------------

def get_lastfm_data(track: str, artist: str) -> Dict:
    url = "http://ws.audioscrobbler.com/2.0/"
    params = {
        "method": "track.getInfo",
        "api_key": LASTFM_API_KEY,
        "artist": artist,
        "track": track,
        "format": "json"
    }

    try:
        r = requests.get(url, params=params, timeout=5)
        data = r.json()

        track_data = data.get("track", {})
        tags = [t["name"] for t in track_data.get("toptags", {}).get("tag", [])]

        return {
            "listeners": track_data.get("listeners"),
            "global_playcount": track_data.get("playcount"),
            "tags": ",".join(tags)
        }
    except Exception:
        return {
            "listeners": None,
            "global_playcount": None,
            "tags": None
        }

# -----------------------------
# CONCURRENT ENRICHMENT (UNIQUE ONLY)
# -----------------------------

def enrich_unique(df: pd.DataFrame) -> pd.DataFrame:
    results = [None] * len(df)

    def task(i, row):
        data = get_lastfm_data(
            row["master_metadata_track_name"],
            row["master_metadata_album_artist_name"]
        )
        return i, data

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(task, i, row)
            for i, row in df.iterrows()
        ]

        for future in as_completed(futures):
            i, data = future.result()
            results[i] = data

    df["listeners"] = [r["listeners"] for r in results]
    df["global_playcount"] = [r["global_playcount"] for r in results]
    df["tags"] = [r["tags"] for r in results]

    return df

# -----------------------------
# JOIN BACK
# -----------------------------

def join_back(agg: pd.DataFrame, enriched: pd.DataFrame) -> pd.DataFrame:
    return agg.merge(
        enriched,
        on=[
            "track_id",
            "master_metadata_track_name",
            "master_metadata_album_artist_name"
        ],
        how="left"
    )

# -----------------------------
# SAVE PARQUET
# -----------------------------

def save_parquet(df: pd.DataFrame, path: str):
    df.to_parquet(path, index=False)

# -----------------------------
# MAIN
# -----------------------------

def main():
    df = load_data(INPUT_CSV)
    df = filter_tracks(df)

    agg, unique_tracks = preprocess(df)

    print(f"Unique tracks to enrich: {len(unique_tracks)}")

    enriched_unique = enrich_unique(unique_tracks)

    final_df = join_back(agg, enriched_unique)

    save_parquet(final_df, OUTPUT_PARQUET)

    print("Pipeline completed. Saved as parquet.")

if __name__ == "__main__":
    main()