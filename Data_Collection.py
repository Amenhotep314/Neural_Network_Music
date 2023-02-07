import spotipy
import json
import os
import subprocess

import basic_pitch
import miditok


with open("spotify.cfg") as credentials:

    content = credentials.readlines()
    SPOTIPY_CLIENT_ID = content[0]
    SPOTIPY_CLIENT_SECRET = content[1]
    SPOTIPY_REDIRECT_URI = content[2]
    username = content[3]


def main():

    token = spotipy.util.prompt_for_user_token(username)

if __name__ == "__main__":

    main()