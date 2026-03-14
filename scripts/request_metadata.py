import requests

client_id = "dd8d1b00e817485686a843374e31824c"
client_secret = "6bbe03a196574438be86e526879bb2b7"

# Get token
auth = requests.post(
    "https://accounts.spotify.com/api/token",
    data={"grant_type": "client_credentials"},
    auth=(client_id, client_secret)
)
print(auth.json())
token = auth.json()["access_token"]

# Use token
headers = {"Authorization": f"Bearer {token}"}
print(headers)
r = requests.get(
    "https://api.spotify.com/v1/audio-features/11dFghVXANMlKmJXsNCbNl",
    headers=headers
)

print(r.json())
