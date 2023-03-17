import os
import numpy as np
import pandas as pd
import librosa
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import filedialog
import pygame

'''
Installations Required:
- python
- librosa
    - pip install librosa
- pygame
    - python3 -m pip install pygame
'''

# Define a function to extract features from audio file
def extract_features(file):
    audio_data, sample_rate = librosa.load(file)
    #n_mfcc = number of samples taken from feature.mfcc
    mfccs = librosa.feature.mfcc(y = audio_data, sr = sample_rate, n_mfcc = 40)
    features = np.mean(mfccs.T, axis=0)
    return features

# Define the GUI function
def select_file():
    # Open a file dialog to select a .wav file
    file_path = filedialog.askopenfilename(filetypes = [('Waveform Audio File Format', '*.wav')])
    # Check if a file was selected
    if file_path:
        # Extract features from the selected file
        features = extract_features(file_path)
        # Make a prediction using the trained classifier
        label = clf.predict([features])[0]
        # Print the label and accuracy to the GUI
        label_text.set("Label: {}".format(label))
        accuracy_text.set("Accuracy: {:.2f}%".format(accuracy*100))
        # Load and play the selected audio file
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

# Create the main window of the GUI
window = tk.Tk()
window.title("ML Audio Classifier - CSS484")

# Create a button to select a file
button = tk.Button(window, text = "Select File", command = select_file)
button.pack()

# Create a button to play the selected audio file
play_button = tk.Button(window, text = "Play", command = lambda: pygame.mixer.music.play())
play_button.pack()

# Create label and accuracy variables for the GUI
label_text = tk.StringVar()
label_text.set("Label: ")
label_label = tk.Label(window, textvariable = label_text)
label_label.pack()

accuracy_text = tk.StringVar()
accuracy_text.set("Accuracy: ")
accuracy_label = tk.Label(window, textvariable = accuracy_text)
accuracy_label.pack()

# Define the file directories and labels
file_dirs = ['/Users/bastien/Desktop/CSS484 - Multimedia Data Processing/Project 4/audio/Speech',
             '/Users/bastien/Desktop/CSS484 - Multimedia Data Processing/Project 4/audio/Music']
labels = ['speech', 'music']

# Read in the audio files and extract features
features = []
for i, file_dir in enumerate(file_dirs):
    label = labels[i]
    files = os.listdir(file_dir)
    for file in files:
        if file.endswith('.wav'):
            file_path = os.path.join(file_dir, file)
            feature = extract_features(file_path)
            features.append([feature, label, file])

# Convert the features to a pandas dataframe
df = pd.DataFrame(features, columns=['feature', 'label', 'file'])
print(df)
print()

# Split the data into training and testing sets
X = np.array(df['feature'].tolist())
y = np.array(df['label'].tolist())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Train a decision tree classifier on the training data
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Initialize pygame.mixer
pygame.mixer.init()

# Start the GUI main loop
# set size of initial frame
greeting = tk.Label(text = "Welcome to the ML Audio Classification System")
greeting.pack()
title = tk.Label(text = "Produced by Thomas Ampalloor, Furqan Kassa, and Bastien Orbigo")
title.pack()
window.geometry("500x200")
window.mainloop()
