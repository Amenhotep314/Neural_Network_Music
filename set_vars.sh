readarray -t lines < spotify.cfg
export SPOTIFY_CLIENT_ID=${lines[0]}
export SPOTIFY_CLIENT_SECRET=${lines[1]}
export SPOTIFY_REDIRECT_URI=${lines[2]}