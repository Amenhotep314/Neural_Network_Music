import spotipy
import spotipy_random
import json
import requests
import os
import random
import string

import basic_pitch
# import miditok


with open("spotify.cfg") as credentials:

    content = credentials.readlines()
    # SPOTIPY_CLIENT_ID = content[0]
    # SPOTIPY_CLIENT_SECRET = content[1]
    # SPOTIPY_REDIRECT_URI = content[2]
    username = content[3]


def main():

    token = spotipy.util.prompt_for_user_token(username)
    spotify = spotipy.Spotify(auth=token)

    alphabet = "abcdefghijklmnopqrstuvwxyz"

    samples = 0
    for letter in alphabet:
        for query in [f"%{letter}", f"{letter}%", f"%{letter}%"]:
            for i in range(20):
                search = spotify.search(query, limit=50, offset=i*50)
                if len(search["tracks"]["items"]) != 50:
                    continue
                for j in range(8):
                    track = search["tracks"]["items"][random.randint(0, len(search["tracks"]["items"])-1)]
                    if (not track["explicit"]) and get_preview_from_track(track):
                        print(samples)
                        samples += 1
                    else:
                        j -= 1

        token = spotipy.util.prompt_for_user_token(username)
        spotify = spotipy.Spotify(auth=token)

def get_preview_from_track(track):

    try:
        url = track["preview_url"]
    except:
        return False

    if track["explicit"] or track["popularity"] == 0:
        return False

    name = track["name"]
    name = name.replace(" ", "_")
    name = name.replace("/", "_")
    name = name.replace(":", "_")
    name = str(track["popularity"]) + "_" + name + ".mp3"

    existing = os.listdir("Tracks")
    if name in existing:
        return False
    print(name)

    web_file = requests.get(url, allow_redirects=True)

    with open(os.path.join("Tracks", name), 'wb') as local_file:
        local_file.write(web_file.content)

    return True


if __name__ == "__main__":

        main()