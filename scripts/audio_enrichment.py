import pandas as pd
import requests
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv


load_dotenv()

# ============================================================
# CONFIG
# ============================================================
INPUT_PARQUET = "../data/silver/1.enriched_tracks.parquet"
OUTPUT_PARQUET = "../data/silver/tracks_with_audio_features.parquet"

CYANITE_API_KEY = os.getenv("CYANITE_TOKEN")
MAX_WORKERS = 10

# ============================================================
# LOAD DATA
# ============================================================

def load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

# ============================================================
# CYANITE AUTH HEADER
# ============================================================

def get_headers():
    return {
        "Authorization": f"Bearer {CYANITE_API_KEY}",
        "Content-Type": "application/json"
    }

# ============================================================
# STEP 1: SEARCH TRACK IN CYANITE
# (We don't have Cyanite IDs, only Spotify metadata)
# ============================================================

def search_track(track: str, artist: str):
    url = "https://api.cyanite.ai/v1/search"

    payload = {
        "query": f"{track} {artist}",
        "limit": 1
    }

    try:
        r = requests.post(url, headers=get_headers(), json=payload, timeout=10)
        print(r)
        data = r.json()

        print(data)
        if "results" in data and len(data["results"]) > 0:
            return data["results"][0]["id"]
        return None
    
    

    except Exception:
        return None

# ============================================================
# STEP 2: GET AUDIO FEATURES FROM CYANITE
# ============================================================

def get_audio_features(cyanite_id: str):
    url = f"https://api.cyanite.ai/v1/tracks/{cyanite_id}"

    try:
        r = requests.get(url, headers=get_headers(), timeout=10)
        data = r.json()

        # Extract useful audio data
        analysis = data.get("analysis", {})

        return {
            "bpm": analysis.get("bpm"),
            "energy": analysis.get("energy"),
            "mood_happy": analysis.get("mood", {}).get("happy"),
            "mood_sad": analysis.get("mood", {}).get("sad"),
            "mood_aggressive": analysis.get("mood", {}).get("aggressive"),
            "mood_relaxed": analysis.get("mood", {}).get("relaxed"),
            "genre": analysis.get("genre"),
            "acousticness_proxy": analysis.get("acousticness")
        }

    except Exception:
        return {
            "bpm": None,
            "energy": None,
            "mood_happy": None,
            "mood_sad": None,
            "mood_aggressive": None,
            "mood_relaxed": None,
            "genre": None,
            "acousticness_proxy": None
        }

# ============================================================
# STEP 3: PIPELINE TASK
# ============================================================

def process_row(i, row):
    track = row["master_metadata_track_name"]
    artist = row["master_metadata_album_artist_name"]

    cyanite_id = search_track(track, artist)

    if cyanite_id is None:
        return i, None

    features = get_audio_features(cyanite_id)

    return i, features

# ============================================================
# STEP 4: CONCURRENT PIPELINE
# ============================================================

def enrich_audio_features(df: pd.DataFrame) -> pd.DataFrame:

    results = [None] * len(df)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = [executor.submit(process_row, i, row) for i, row in df.iterrows()]

        for future in as_completed(futures):
            i, data = future.result()
            results[i] = data

    df["bpm"] = [r["bpm"] if r else None for r in results]
    df["energy"] = [r["energy"] if r else None for r in results]
    df["mood_happy"] = [r["mood_happy"] if r else None for r in results]
    df["mood_sad"] = [r["mood_sad"] if r else None for r in results]
    df["mood_aggressive"] = [r["mood_aggressive"] if r else None for r in results]
    df["mood_relaxed"] = [r["mood_relaxed"] if r else None for r in results]
    df["genre_audio"] = [r["genre"] if r else None for r in results]
    df["acousticness_proxy"] = [r["acousticness_proxy"] if r else None for r in results]

    return df

# ============================================================
# SAVE
# ============================================================

def save_parquet(df: pd.DataFrame, path: str):
    df.to_parquet(path, index=False)

# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading silver dataset...")
    df = load_data(INPUT_PARQUET)

    print(f"Tracks to enrich with audio features: {len(df)}")

    df = enrich_audio_features(df)

    print("Saving dataset with audio features...")
    save_parquet(df, OUTPUT_PARQUET)

    print("Done.")


if __name__ == "__main__":
    main()
