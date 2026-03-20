import pandas as pd
import numpy as np
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

HISTORY_CSV = BASE_DIR / "data/bronze/spotify_history.csv"
ENRICHED_PARQUET = BASE_DIR / "data/silver/1.enriched_tracks.parquet"
KAGGLE_CSV = BASE_DIR / "data/bronze/audio_features_kaggle.csv"

OUTPUT_DIR = BASE_DIR / "data/gold"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── GENRE MAPPING ──────────────────────────────────────────
# Manual mapping for the most-played artists
ARTIST_GENRE = {
    "Maroon 5": "Pop Rock",
    "Shinedown": "Hard Rock",
    "Muse": "Alternative Rock",
    "Leprous": "Progressive Metal",
    "Imagine Dragons": "Pop Rock",
    "Sonata Arctica": "Power Metal",
    "Billie Eilish": "Pop",
    "The Pineapple Thief": "Progressive Rock",
    "Thirty Seconds To Mars": "Alternative Rock",
    "Porcupine Tree": "Progressive Rock",
    "Avenged Sevenfold": "Hard Rock",
    "Dream Theater": "Progressive Metal",
    "The Rasmus": "Alternative Rock",
    "Fito Paez": "Latin Rock",
    "IAMX": "Electronic",
    "Steven Wilson": "Progressive Rock",
    "Queen": "Classic Rock",
    "Pink Floyd": "Progressive Rock",
    "Ed Sheeran": "Pop",
    "Arctic Monkeys": "Indie Rock",
    "FINNEAS": "Pop",
    "Opeth": "Progressive Metal",
    "Taylor Swift": "Pop",
    "Ariana Grande": "Pop",
    "Robbie Williams": "Pop Rock",
    "The Dear Hunter": "Progressive Rock",
    "blink-182": "Punk Rock",
    "In This Moment": "Metal",
    "Miley Cyrus": "Pop",
    "Frédéric Chopin": "Classical",
    "Radiohead": "Alternative Rock",
    "Linkin Park": "Alternative Rock",
    "Green Day": "Punk Rock",
    "The Offspring": "Punk Rock",
    "Depeche Mode": "Electronic",
    "Foo Fighters": "Alternative Rock",
    "Nirvana": "Grunge",
    "Red Hot Chili Peppers": "Alternative Rock",
    "Metallica": "Metal",
    "Iron Maiden": "Metal",
    "Slipknot": "Metal",
    "System Of A Down": "Metal",
    "Korn": "Metal",
    "Tool": "Progressive Metal",
    "Deftones": "Alternative Metal",
    "Haken": "Progressive Metal",
    "Nightwish": "Symphonic Metal",
    "Epica": "Symphonic Metal",
    "Kamelot": "Symphonic Metal",
    "Arch Enemy": "Metal",
    "Children Of Bodom": "Metal",
    "Megadeth": "Metal",
    "Stratovarius": "Power Metal",
    "Blind Guardian": "Power Metal",
    "Rhapsody Of Fire": "Power Metal",
    "The Weeknd": "R&B",
    "Dua Lipa": "Pop",
    "Lady Gaga": "Pop",
    "Katy Perry": "Pop",
    "Beyoncé": "Pop",
    "Rihanna": "Pop",
    "Bruno Mars": "Pop",
    "Post Malone": "Hip Hop",
    "Eminem": "Hip Hop",
    "Kendrick Lamar": "Hip Hop",
    "Drake": "Hip Hop",
    "Kanye West": "Hip Hop",
    "Travis Scott": "Hip Hop",
    "Hans Zimmer": "Soundtrack",
    "Howard Shore": "Soundtrack",
    "John Williams": "Soundtrack",
    "Ramin Djawadi": "Soundtrack",
    "Gustavo Santaolalla": "Soundtrack",
    "Ludwig van Beethoven": "Classical",
    "Johann Sebastian Bach": "Classical",
    "Franz Schubert": "Classical",
    "Wolfgang Amadeus Mozart": "Classical",
    "Joseph Haydn": "Classical",
    "Soda Stereo": "Latin Rock",
    "Gustavo Cerati": "Latin Rock",
    "Ciro y los Persas": "Latin Rock",
    "Las Pastillas del Abuelo": "Latin Rock",
    "Bersuit Vergarabat": "Latin Rock",
    "Eruca Sativa": "Latin Rock",
    "No Te Va Gustar": "Latin Rock",
    "Patricio Rey y sus Redonditos de Ricota": "Latin Rock",
    "La Renga": "Latin Rock",
    "Callejeros": "Latin Rock",
    "Abel Pintos": "Latin Pop",
    "Roberto Carlos": "Latin Pop",
    "The Beatles": "Classic Rock",
    "Led Zeppelin": "Classic Rock",
    "The Rolling Stones": "Classic Rock",
    "AC/DC": "Classic Rock",
    "Bon Jovi": "Classic Rock",
    "Guns N' Roses": "Classic Rock",
    "Aerosmith": "Classic Rock",
    "The Who": "Classic Rock",
    "Deep Purple": "Classic Rock",
    "Black Sabbath": "Classic Rock",
    "David Bowie": "Classic Rock",
    "Genesis": "Progressive Rock",
    "Yes": "Progressive Rock",
    "Rush": "Progressive Rock",
    "Gentle Giant": "Progressive Rock",
    "Camel": "Progressive Rock",
    "Skrillex": "Electronic",
    "Tiësto": "Electronic",
    "Deadmau5": "Electronic",
    "M83": "Electronic",
    "Daft Punk": "Electronic",
    "The Prodigy": "Electronic",
    "Modeselektor": "Electronic",
    "Boards of Canada": "Electronic",
    "Circa Survive": "Post-Hardcore",
    "Rise Against": "Punk Rock",
    "Sum 41": "Punk Rock",
    "NOFX": "Punk Rock",
    "Bad Religion": "Punk Rock",
    "Anti-Flag": "Punk Rock",
    "Ramones": "Punk Rock",
    "Panic! At The Disco": "Pop Rock",
    "Fall Out Boy": "Pop Rock",
    "Paramore": "Pop Rock",
    "Twenty One Pilots": "Pop Rock",
    "Coldplay": "Pop Rock",
    "OneRepublic": "Pop Rock",
    "The Script": "Pop Rock",
    "Bastille": "Pop Rock",
    "Placebo": "Alternative Rock",
    "Cage The Elephant": "Alternative Rock",
    "Arcade Fire": "Indie Rock",
    "The Killers": "Indie Rock",
    "Franz Ferdinand": "Indie Rock",
    "Kasabian": "Indie Rock",
    "Interpol": "Indie Rock",
    "Travis": "Indie Rock",
    "Oasis": "Indie Rock",
    "Royal Blood": "Alternative Rock",
    "Lana Del Rey": "Pop",
    "Lorde": "Pop",
    "Sam Fender": "Indie Rock",
    "Nick Jonas": "Pop",
    "ZAYN": "Pop",
    "Backstreet Boys": "Pop",
    "Britney Spears": "Pop",
    "Rick Astley": "Pop",
    "ABBA": "Pop",
    "George Michael": "Pop",
    "Savage Garden": "Pop",
    "Will Smith": "Hip Hop",
    "Thomas Bergersen": "Soundtrack",
    "Two Steps from Hell": "Soundtrack",
    "Lorne Balfe": "Soundtrack",
    "Alexandre Desplat": "Soundtrack",
    "James Newton Howard": "Soundtrack",
    "Dario Marianelli": "Soundtrack",
    "Trent Reznor": "Electronic",
    "Seether": "Hard Rock",
    "Godsmack": "Hard Rock",
    "Alter Bridge": "Hard Rock",
    "Hoobastank": "Hard Rock",
    "Saliva": "Hard Rock",
    "Three Days Grace": "Hard Rock",
    "Breaking Benjamin": "Hard Rock",
    "Disturbed": "Hard Rock",
    "Evanescence": "Hard Rock",
    "Bring Me The Horizon": "Metal",
    "Poets of the Fall": "Alternative Rock",
    "Hurts": "Electronic",
    "Chet Faker": "Electronic",
    "Boyce Avenue": "Pop Rock",
    "James Bay": "Pop",
    "JP Saxe": "Pop",
    "Ben Platt": "Pop",
    "Luciano Pavarotti": "Classical",
    "Tony Bennett": "Jazz",
    "John Coltrane Quartet": "Jazz",
    "B.B. King": "Blues",
    "Elliott Smith": "Indie Rock",
    "Blind Melon": "Alternative Rock",
    "The Stooges": "Classic Rock",
    "Iggy Pop": "Classic Rock",
    "Peter Gabriel": "Progressive Rock",
    "Supergrass": "Indie Rock",
    "Pixies": "Alternative Rock",
    "The Smashing Pumpkins": "Alternative Rock",
    "Nickel Creek": "Folk",
    "Lisa Loeb": "Folk",
    "Aquilo": "Electronic",
    "Lenny Kravitz": "Classic Rock",
    "Avril Lavigne": "Pop Rock",
    "Michael Jackson": "Pop",
    "Chris Cornell": "Grunge",
    "Soundgarden": "Grunge",
    "Alice In Chains": "Grunge",
    "Pearl Jam": "Grunge",
    "Onda Vaga": "Latin Folk",
    "La Vida Bohème": "Latin Rock",
    "Connor Questa": "Latin Rock",
    "Zambayonny": "Latin Rock",
    "Karina": "Latin Pop",
    "Thalia": "Latin Pop",
    "Romeo Santos": "Latin Pop",
    "Ricky Martin": "Latin Pop",
    "Alejandro Sanz": "Latin Pop",
    "Transatlantic": "Progressive Rock",
    "Dredg": "Alternative Rock",
    "VersaEmerge": "Post-Hardcore",
    "Thank You Scientist": "Progressive Rock",
    "Darkwater": "Progressive Metal",
    "Scale The Summit": "Progressive Metal",
    "Nevermore": "Progressive Metal",
    "Klone": "Progressive Metal",
    "Barns Courtney": "Indie Rock",
    "Magic Sword": "Electronic",
    "Foreigner": "Classic Rock",
    "Steely Dan": "Classic Rock",
    "Poison": "Classic Rock",
    "Enuff Z'Nuff": "Classic Rock",
    "Mr. Big": "Classic Rock",
    "Van Canto": "Power Metal",
    "Majestica": "Power Metal",
    "Edenbridge": "Symphonic Metal",
    "Klimt 1918": "Post-Rock",
    "Saturnus": "Doom Metal",
    "Yann Tiersen": "Soundtrack",
    "Richard Galliano": "Jazz",
    "Beck": "Alternative Rock",
    "X Ambassadors": "Pop Rock",
    "Macklemore & Ryan Lewis": "Hip Hop",
    "Mark Ronson": "Pop",
    "Miranda Lambert": "Country",
    "Famous Last Words": "Post-Hardcore",
    "Lindemann": "Metal",
    "Nine Lashes": "Hard Rock",
    "All Good Things": "Hard Rock",
    "Adema": "Hard Rock",
    "Les Friction": "Soundtrack",
    "Donots": "Punk Rock",
    "Ends With A Bullet": "Hard Rock",
    "Patrick Stump": "Pop Rock",
    "Paul Young": "Pop",
    "Allie X": "Pop",
    "Nessa Barrett": "Pop",
    "VÉRITÉ": "Pop",
    "Boy Epic": "Electronic",
    "Heatbeat": "Electronic",
    "Miami Nights 1984": "Electronic",
    "Oneohtrix Point Never": "Electronic",
    "Facundo Toro": "Latin Folk",
    "La Reserva": "Latin Rock",
    "Barco": "Latin Rock",
    "Pork": "Latin Rock",
    "JAF": "Latin Rock",
}

TAG_TO_GENRE = {
    "rock": "Rock",
    "indie rock": "Indie Rock",
    "indie": "Indie Rock",
    "alternative": "Alternative Rock",
    "alternative rock": "Alternative Rock",
    "pop": "Pop",
    "pop rock": "Pop Rock",
    "classic rock": "Classic Rock",
    "metal": "Metal",
    "heavy metal": "Metal",
    "progressive rock": "Progressive Rock",
    "prog rock": "Progressive Rock",
    "progressive metal": "Progressive Metal",
    "prog metal": "Progressive Metal",
    "punk": "Punk Rock",
    "punk rock": "Punk Rock",
    "electronic": "Electronic",
    "synthpop": "Electronic",
    "edm": "Electronic",
    "hip-hop": "Hip Hop",
    "hip hop": "Hip Hop",
    "rap": "Hip Hop",
    "rnb": "R&B",
    "r&b": "R&B",
    "soul": "R&B",
    "jazz": "Jazz",
    "blues": "Blues",
    "folk": "Folk",
    "country": "Country",
    "classical": "Classical",
    "soundtrack": "Soundtrack",
    "latin": "Latin Pop",
    "reggaeton": "Latin Pop",
    "dance": "Electronic",
    "grunge": "Grunge",
    "post-rock": "Post-Rock",
    "post-hardcore": "Post-Hardcore",
    "power metal": "Power Metal",
    "symphonic metal": "Symphonic Metal",
    "doom metal": "Doom Metal",
    "hard rock": "Hard Rock",
}


def extract_track_id(uri):
    if pd.isna(uri):
        return None
    match = re.search(r"track:([a-zA-Z0-9]+)", uri)
    return match.group(1) if match else None


def genre_from_tags(tags_str):
    if pd.isna(tags_str) or tags_str.strip() == "":
        return None
    tags = [t.strip().lower() for t in tags_str.split(",")]
    for tag in tags:
        if tag in TAG_TO_GENRE:
            return TAG_TO_GENRE[tag]
    return None


def assign_genre(row):
    # 1. Try artist mapping first (most reliable)
    artist = row["master_metadata_album_artist_name"]
    if artist in ARTIST_GENRE:
        return ARTIST_GENRE[artist]
    # 2. Try Last.fm tags
    tag_genre = genre_from_tags(row.get("tags", ""))
    if tag_genre:
        return tag_genre
    return "Other"


def load_history():
    df = pd.read_csv(HISTORY_CSV, sep=";", low_memory=False)
    df = df[df["master_metadata_track_name"].notna()]
    df = df[df["spotify_track_uri"].str.contains("track:", na=False)]
    df["track_id"] = df["spotify_track_uri"].apply(extract_track_id)
    df = df[df["track_id"].notna()]

    df["ts"] = pd.to_datetime(df["ts"])
    df["year"] = df["ts"].dt.year
    df["month"] = df["ts"].dt.month
    df["year_month"] = df["ts"].dt.to_period("M").astype(str)
    df["day_of_week"] = df["ts"].dt.day_name()
    df["hour"] = df["ts"].dt.hour
    df["minutes_played"] = (df["ms_played"] / 60000).round(2)

    return df


def load_enriched():
    return pd.read_parquet(ENRICHED_PARQUET)


def load_kaggle_features():
    kaggle = pd.read_csv(KAGGLE_CSV, low_memory=False)
    kaggle = kaggle.rename(columns={"track_id": "track_id"})
    feature_cols = [
        "track_id", "danceability", "energy", "key", "loudness",
        "speechiness", "acousticness", "instrumentalness",
        "liveness", "valence", "tempo"
    ]
    return kaggle[feature_cols].drop_duplicates(subset="track_id")


def build_gold():
    print("Loading history...")
    history = load_history()
    print(f"  {len(history)} play events")

    print("Loading enriched tracks...")
    enriched = load_enriched()
    print(f"  {len(enriched)} unique tracks")

    print("Assigning genres...")
    enriched["genre"] = enriched.apply(assign_genre, axis=1)
    genre_counts = enriched["genre"].value_counts()
    print(f"  Genre distribution:")
    for g, c in genre_counts.items():
        print(f"    {g}: {c}")

    print("Loading Kaggle audio features...")
    kaggle = load_kaggle_features()
    print(f"  {len(kaggle)} tracks in Kaggle dataset")

    enriched = enriched.merge(kaggle, on="track_id", how="left")
    has_features = enriched["danceability"].notna().sum()
    print(f"  Matched: {has_features} / {len(enriched)} ({has_features/len(enriched)*100:.1f}%)")

    # Mock audio features for tracks without them
    print("Mocking audio features for remaining tracks...")
    rng = np.random.default_rng(42)
    for col in ["danceability", "energy", "valence", "acousticness"]:
        mask = enriched[col].isna()
        enriched.loc[mask, col] = rng.beta(2, 2, size=mask.sum()).round(3)
    mask = enriched["tempo"].isna()
    enriched.loc[mask, "tempo"] = rng.normal(120, 25, size=mask.sum()).clip(60, 220).round(1)

    # Build play-level gold dataset
    print("Building gold dataset...")
    track_meta = enriched[[
        "track_id", "master_metadata_track_name", "master_metadata_album_artist_name",
        "total_ms", "play_count", "listeners", "global_playcount", "tags", "genre",
        "danceability", "energy", "valence", "acousticness", "tempo"
    ]].rename(columns={
        "master_metadata_track_name": "track_name",
        "master_metadata_album_artist_name": "artist",
    })

    gold = history[[
        "ts", "track_id", "master_metadata_track_name", "master_metadata_album_artist_name",
        "master_metadata_album_album_name", "ms_played", "minutes_played",
        "platform", "reason_start", "reason_end", "shuffle", "skipped",
        "year", "month", "year_month", "day_of_week", "hour"
    ]].rename(columns={
        "master_metadata_track_name": "track_name",
        "master_metadata_album_artist_name": "artist",
        "master_metadata_album_album_name": "album",
    })

    gold = gold.merge(
        track_meta[["track_id", "genre", "play_count", "listeners", "global_playcount",
                     "danceability", "energy", "valence", "acousticness", "tempo"]],
        on="track_id",
        how="left"
    )
    gold["genre"] = gold["genre"].fillna("Other")

    print(f"  Gold dataset: {gold.shape}")

    # ─── EXPORT: Full gold ───────────────────────────────────
    gold_path = OUTPUT_DIR / "full_history.csv"
    gold.to_csv(gold_path, index=False)
    print(f"  Saved: {gold_path}")

    # ─── EXPORT: RAWGraphs ───────────────────────────────────
    # Alluvial / Sankey: genre evolution across years
    rawgraph = gold.groupby(["year", "genre"]).agg(
        play_count=("track_id", "count"),
        total_minutes=("minutes_played", "sum")
    ).reset_index()
    rawgraph["total_minutes"] = rawgraph["total_minutes"].round(1)
    rawgraph_path = OUTPUT_DIR / "rawgraph_genre_by_year.csv"
    rawgraph.to_csv(rawgraph_path, index=False)
    print(f"  Saved: {rawgraph_path}")

    # RAWGraphs: artist-genre treemap
    treemap = gold.groupby(["genre", "artist"]).agg(
        play_count=("track_id", "count"),
        total_minutes=("minutes_played", "sum")
    ).reset_index()
    treemap["total_minutes"] = treemap["total_minutes"].round(1)
    treemap = treemap[treemap["play_count"] >= 5]
    treemap_path = OUTPUT_DIR / "rawgraph_artist_treemap.csv"
    treemap.to_csv(treemap_path, index=False)
    print(f"  Saved: {treemap_path}")

    # ─── EXPORT: Flourish ────────────────────────────────────
    # Bar chart race: top artists by cumulative plays per year
    artist_year = gold.groupby(["year", "artist"]).agg(
        plays=("track_id", "count"),
        minutes=("minutes_played", "sum")
    ).reset_index()
    artist_year["minutes"] = artist_year["minutes"].round(1)

    # Pivot for Flourish bar chart race format (years as columns, artists as rows)
    cumulative = artist_year.pivot_table(
        index="artist", columns="year", values="plays", fill_value=0
    )
    cumulative = cumulative.cumsum(axis=1)
    # Keep only artists with at least 50 total plays
    cumulative = cumulative[cumulative.iloc[:, -1] >= 50]
    cumulative = cumulative.reset_index()
    cumulative.columns = [str(c) for c in cumulative.columns]
    flourish_race_path = OUTPUT_DIR / "flourish_bar_race.csv"
    cumulative.to_csv(flourish_race_path, index=False)
    print(f"  Saved: {flourish_race_path}")

    # Flourish scatter: tracks by audio features
    scatter = track_meta[["track_name", "artist", "genre", "play_count",
                          "danceability", "energy", "valence", "tempo"]].copy()
    scatter = scatter[scatter["play_count"] >= 3]
    flourish_scatter_path = OUTPUT_DIR / "flourish_scatter.csv"
    scatter.to_csv(flourish_scatter_path, index=False)
    print(f"  Saved: {flourish_scatter_path}")

    # ─── EXPORT: Tableau ─────────────────────────────────────
    # Full denormalized dataset with all dimensions
    tableau_path = OUTPUT_DIR / "tableau_full.csv"
    gold.to_csv(tableau_path, index=False)
    print(f"  Saved: {tableau_path}")

    # Tableau summary: monthly aggregation
    monthly = gold.groupby(["year", "month", "year_month", "genre"]).agg(
        plays=("track_id", "count"),
        unique_tracks=("track_id", "nunique"),
        unique_artists=("artist", "nunique"),
        total_minutes=("minutes_played", "sum"),
        avg_danceability=("danceability", "mean"),
        avg_energy=("energy", "mean"),
        avg_valence=("valence", "mean"),
    ).reset_index()
    monthly["total_minutes"] = monthly["total_minutes"].round(1)
    monthly["avg_danceability"] = monthly["avg_danceability"].round(3)
    monthly["avg_energy"] = monthly["avg_energy"].round(3)
    monthly["avg_valence"] = monthly["avg_valence"].round(3)
    tableau_monthly_path = OUTPUT_DIR / "tableau_monthly.csv"
    monthly.to_csv(tableau_monthly_path, index=False)
    print(f"  Saved: {tableau_monthly_path}")

    print("\nDone. All exports in data/gold/")


if __name__ == "__main__":
    build_gold()
