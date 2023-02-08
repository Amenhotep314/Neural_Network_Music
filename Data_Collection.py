import spotipy
import spotipy_random
import json
import requests
import os
import random
import string

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

    samples = 0
    while samples < 10000:
        track = spotipy_random.get_random(spotify=spotify, type="track")
        if (not track["explicit"]) and get_preview_from_track(track):
            print(samples)
            samples += 1

    # print(json.dumps(spotify.category_playlists("toplists"), sort_keys=True, indent=4))



def get_preview_from_track(track):

    try:
        url = track["preview_url"]
    except:
        return False

    web_file = requests.get(url, allow_redirects=True)
    name = str(track["popularity"]) + "_" + track["name"].replace(" ", "_") + ".mp3"

    existing = os.listdir("Tracks")
    if name in existing:
        return False

    print(name)
    with open(os.path.join("Tracks", name), 'wb') as local_file:
        local_file.write(web_file.content)

    return True


if __name__ == "__main__":

        main()