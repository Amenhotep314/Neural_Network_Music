readarray -t lines < spotify.cfg
export SPOTIPY_CLIENT_ID=${lines[0]}
export SPOTIPY_CLIENT_SECRET=${lines[1]}
export SPOTIPY_REDIRECT_URI=${lines[2]}