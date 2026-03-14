import json
import pandas as pd
from pathlib import Path

# Folder containing the JSON files
json_folder = Path("data")

# List to store all rows
records = []

# Read every JSON file in the folder
for file in json_folder.glob("*.json"):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

        # If the file is a list of records
        if isinstance(data, list):
            records.extend(data)
        else:
            records.append(data)

# Convert to DataFrame
df = pd.DataFrame(records)

# Export to CSV
df.to_csv("spotify_history.csv", index=False)

print("CSV created: spotify_history.csv")
