import requests
import os 
from dotenv import load_dotenv
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd


load_dotenv()

RECCO_KEY = os.getenv('ROCCO_KEY')

BASE_URL = "https://api.reccobeats.com/v1/track/id:{}/audio-features"


def get_track_audio_features(track_id, session):
    headers = {"Authorization": f"Bearer {RECCO_KEY}"}

    try:
        r = session.get(BASE_URL.format(track_id), headers=headers, timeout=10)

        if r.status_code == 200:
            data = r.json()

            return {
                "track_id": track_id,
                "danceability": data["content"]["danceability"],
                "energy": data["content"]["energy"],
                "tempo": data["content"]["tempo"],
                "valence": data["content"]["valence"]
            }

        else:
            return {"track_id": track_id, "error": r.status_code}

    except Exception as e:
        return {"track_id": track_id, "error": str(e)}


def get_multiple_tracks(track_ids, max_workers=10):

    results = []

    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:

            futures = {
                executor.submit(get_track_audio_features, tid, session): tid
                for tid in track_ids
            }

            for future in as_completed(futures):
                results.append(future.result())

    return results

tracks = pd.read_csv("../data/track_ids.csv")

df = pd.DataFrame(get_multiple_tracks(tracks['tracks']))
