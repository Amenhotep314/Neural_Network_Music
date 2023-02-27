"""Data_Collection.py: Contains all functionality for obtaining a data set and preparing it for use in training."""

import requests
import os
import random
import shutil
from pathlib import Path

import spotipy
import miditok
from basic_pitch import inference


folder_names = {
    "mp3": "MP3s",
    "midi": "MIDIs",
    "tok": "Tokens",
    "test": "Test_Data",
    "train": "Training_Data"
}


def main():

    # download_mp3s()
    # convert_mp3s_to_midis()
    # convert_midis_to_tokens()
    # split_dataset()


def download_mp3s():

    """Downloads 30-second MP3 files from the Spotify developer API using open-ended alphabet searches. Expect between 5000 and 6000 downloads."""

    # Reads username (and optionally) application verification details from the spotify.cfg file that each user must create
    with open("spotify.cfg") as credentials:
        content = credentials.readlines()
        # SPOTIPY_CLIENT_ID = content[0]
        # SPOTIPY_CLIENT_SECRET = content[1]
        # SPOTIPY_REDIRECT_URI = content[2]
        username = content[3]

    create_folder("mp3")

    # Sets up access to the Spotify API
    token = spotipy.util.prompt_for_user_token(username)
    spotify = spotipy.Spotify(auth=token)

    alphabet = "abcdefghijklmnopqrstuvwxyz"

    print("Downloading MP3s.")
    print("\t".join(["#", "Popularity", "Title"]))
    # Loops through songs, trying to download them
    samples = 0
    # Loop through the alphabet
    for letter in alphabet:
        # Loop through searches, looking for the current letter in different parts of the title
        for query in [f"%{letter}", f"{letter}%", f"%{letter}%"]:
            # Loop through search pages
            for i in range(20):
                search = spotify.search(query, limit=50, offset=i*50)
                if len(search["tracks"]["items"]) != 50:
                    continue
                # Loop through songs in search
                for j in range(50):
                    track = search["tracks"]["items"][j]
                    if get_mp3_from_title(track):
                        print("\t".join([str(samples), str(track["popularity"]), track["name"]]))
                        samples += 1

        token = spotipy.util.prompt_for_user_token(username)
        spotify = spotipy.Spotify(auth=token)


def get_mp3_from_title(track):

    """Attempts to download a specific 30-second MP3 track from the Spotify developer API.
    Args:
        track: A JSON object representing the track to be sampled
    Returns:
        bool: True if the track was saved, otherwise False"""

    # The track must meet criteria to be included
    # It must have a url
    try:
        url = track["preview_url"]
    except:
        return False

    # It must be clean and have some popularity
    if track["explicit"] or track["popularity"] == 0:
        return False

    # It must not already be downloaded
    name = create_filename_from_title(track)
    existing = os.listdir(folder_names["mp3"])
    if name in existing:
        return False

    # Its url must be accessible
    try:
        web_file = requests.get(url, allow_redirects=True)
    except requests.exceptions.MissingSchema:
        return False

    # If it meets these requirements, save it
    with open(os.path.join(folder_names["mp3"], name), 'wb') as local_file:
        local_file.write(web_file.content)

    return True


def create_filename_from_title(track, extension="mp3"):

    """Prepares a filesystem-safe name for a file.
    Args:
        track: A JSON object representing the track to be sampled
        extension: The extension to be added to the filename (default is mp3)
    Returns:
        str: The name to be saved in the filesystem"""

    name = track["name"]
    name = name.replace(" ", "_")
    name = name.replace("/", "_")
    name = name.replace(":", "_")
    name = str(track["popularity"]) + "_" + name + "." + extension

    return name


def convert_mp3s_to_midis():

    """Takes downloaded mp3s and converts them to midi files using Spotify's basic_pitch process."""

    # Prepares the folder
    create_folder("midi")

    filenames = os.listdir(folder_names["mp3"])
    paths = [os.path.join(folder_names["mp3"], filename) for filename in filenames]

    print("Converting MP3s to MIDIs.")
    inference.predict_and_save(paths, folder_names["midi"], True, False, False, False)

    # The basic_pitch library adds the string "_basic_pitch" to the end of converted files' names. Correct this.
    print("Renaming MIDIs.")

    for midi in os.listdir(folder_names["midi"]):
        print(midi)
        target = os.path.join(folder_names["midi"], midi)
        new = os.path.join(folder_names["midi"], midi.replace("_basic_pitch", ""))
        os.rename(target, new)


def convert_midis_to_tokens(tokenization_method="midilike"):

    """Takes all midi files in the data set, tokenizes them for machine learning, and saves them using JSON.
    Args:
        tokenization_method (str): The name of the MidiTok method to be used. Can be cpword, midilike (default), mumidi, octuple, remi, or structured"""

    # These are simply all of the presents offered by the library
    tokenization_methods = {
        "cpword": miditok.CPWord(),
        "midilike": miditok.MIDILike(),
        "mumidi": miditok.MuMIDI(),
        "octuple": miditok.Octuple(),
        "remi": miditok.REMI(),
        "structured": miditok.Structured()
    }

    # Prepare the folder
    create_folder("tok")

    # Select the method to use and list the paths to all the midi files
    tokenizer = tokenization_methods[tokenization_method]
    midi_paths = list(Path(folder_names["midi"]).glob('**/*.mid'))

    # Tokenize the data. Reduce size by combining common tokens with bpe algorithm
    print("Tokenizing data.")
    tokenizer.tokenize_midi_dataset(midi_paths, Path(folder_names["tok"] + "_tmp"))
    tokenizer.learn_bpe(tokens_path=Path(folder_names["tok"] + "_tmp"), vocab_size=500, out_dir=Path(folder_names["tok"]), files_lim=300)
    tokenizer.apply_bpe_to_dataset(Path(folder_names["tok"] + "_tmp"), Path(folder_names["tok"]))

    # Sort out the files
    shutil.rmtree(folder_names["tok"] + "_tmp")
    os.rename(os.path.join(folder_names["tok"], "config.txt"), "miditok_config.txt")


def split_dataset(train_percentage=0.8):

    """Divides the tokenized dataset into two parts, one used for training and the other for testing.
    Args:
        train_percentage (float): The percentage of the total dataset to be allocated for training the neural network. Default is 0.8"""

    create_folder("train")
    create_folder("test")

    tokens = os.listdir(folder_names["tok"])
    random.shuffle(tokens)
    train_num = int(len(tokens) * train_percentage)
    random.shuffle(tokens)
    train_tokens = tokens[:train_num]
    test_tokens = tokens[train_num:]

    print("Splitting data set.")
    for token in train_tokens:
        shutil.copy(os.path.join(folder_names["tok"], token), os.path.join(folder_names["train"], token))

    for token in test_tokens:
        shutil.copy(os.path.join(folder_names["tok"], token), os.path.join(folder_names["test"], token))


def create_folder(name):

    """Creates a data folder if it doesn't already exist.
    Args:
        name: A string representing the type of data to be stored. Corresponds to the global dict folder_names"""

    path = folder_names[name]
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == "__main__":

        main()