import requests
from pprint import pprint
from settings import API_KEY


def filter_english_speakers(data):
    # List of English locales to filter by
    english_locales = ["en-US", "en-GB", "en-AU"]

    # Filter the data to include only speakers with English locales
    filtered_speakers = [
        {
            "id": speaker["id"],
            "displayName": speaker["displayName"],
            "locale": speaker["locale"],
            "gender": speaker["gender"],
        }
        for speaker in data["data"]
        # if speaker["locale"] in english_locales
        if speaker["locale"].startswith("en-") and len(speaker["speakerStyles"]) > 1
    ]

    return filtered_speakers


def get_speakers():
    url = "https://api.genny.lovo.ai/api/v1/speakers?sort=displayName%3A1"

    headers = {"accept": "application/json", "X-API-KEY": API_KEY}

    data = requests.get(url, headers=headers).json()
    return filter_english_speakers(data)


if __name__ == "__main__":
    speakers = get_speakers()
    pprint(speakers)
