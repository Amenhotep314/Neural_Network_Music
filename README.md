# Neural_Network_Music
This repository contains all software and data associated with my senior capstone research project at St. Johnsbury Academy, *Music Popularity Prediction with Neural Networks: A Composition-Based Approach*. A copy of the paper is also included, and it includes an in-depth review of existing literature on the subject, a rigorous outline of the procedure followed, and a presentation and analysis of the findings.

## Research
This research focuses primarily on the potential of neural networks to predict the popularty of a piece of music on the Spotify streaming service based solely on compositional data. Though much research has been done on the application of neural networks to music, very little exists currently concerning the phenomenon of popularity. In the process applied here, a dataset of MIDI audio samples was created and used to train a convolutional neural network in several iterations. Analysis of these iterations showed general poor performance, and it was concluded that a convolutional neural network trained on compositional data alone is not a reliable predictor of music popularity on streaming services. It was also conjectured that compositional data alone is probably not a very good indicator of probability.

## Use
This software was developed using Anaconda Python 3. Third-party libraries required are listed and can be pipped. In each of the two primary .py scripts here, use the main() functions to specify the dataset creation or model training and evalution operations you'd like to complete.
