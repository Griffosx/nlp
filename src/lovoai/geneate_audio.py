import requests
from random import choice
from settings import API_KEY
from constants import WORDS, SPEAKERS


def get_audio(text: str, speaker: dict, speed: float):
    url = "https://api.genny.lovo.ai/api/v1/tts/sync"

    payload = {
        "speed": speed,
        "text": text,
        "speaker": speaker["id"]
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "X-API-KEY": API_KEY
    }

    # Sending POST request to generate audio
    response = requests.post(url, json=payload, headers=headers)

    # Parse the response to JSON
    response_json = response.json()

    # Extract the audio URL from the response
    audio_url = response_json["data"][0]["urls"][0]

    # Download the audio file
    audio_response = requests.get(audio_url)

    # Save the audio file locally
    speaker_name = speaker["displayName"].lower().replace(" ", "_")
    filename = f"audio/{text}_{speaker_name}_{int(speed * 10)}.wav"
    with open(filename, 'wb') as audio_file:
        audio_file.write(audio_response.content)

    print(f"Audio file saved as {filename}")


if __name__ == "__main__":
    speeds = [0.8, 1, 1.2]

    for word in WORDS:
        for speaker in SPEAKERS[35:40]:
            speed = choice(speeds)
            try:
                get_audio(word, speaker, speed)
            except Exception as e:
                print(f"Impossible to generate audio of {word} by {speaker} with speed {speed}. Exception {e}")
