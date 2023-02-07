import spotipy
import json
import requests
import os
import subprocess

import basic_pitch
import miditok


with open("spotify.cfg") as credentials:

    content = credentials.readlines()
    # SPOTIPY_CLIENT_ID = content[0]
    # SPOTIPY_CLIENT_SECRET = content[1]
    # SPOTIPY_REDIRECT_URI = content[2]
    username = content[3]


def main():

    token = spotipy.util.prompt_for_user_token(username)
    spotify = spotipy.Spotify(auth=token)

    track = spotify.track(input("URI: "))
    print(json.dumps(track, sort_keys=True, indent=4))

    try:
        url = track["preview_url"]
    except:
        print("No preview found.")
        url = ""

    if url:
        web_file = requests.get(url, allow_redirects=True)
        name = str(track["popularity"]) + "_" + track["name"].replace(" ", "_") + ".mp3"
        with open(os.path.join("Tracks", name), 'wb') as local_file:
            local_file.write(web_file.content)

if __name__ == "__main__":

    while True:
        main()