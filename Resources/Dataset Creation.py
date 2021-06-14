import time #to calculate time taken for each genre calculation
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('Data/genres_original/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import librosa # to calculate features from the audio file
import librosa.display
import matplotlib.pyplot as plt
%matplotlib inline
from PIL import Image
import pathlib
import csv # for loading the features into a file for future use
import warnings
warnings.filterwarnings('ignore')


#specifying the header or the attributes of our dataset
header = 'filename length tempo mean(chroma_stft) var(chroma_stft) mean(rms) var(rms) mean(spec_cent) (spec_cent) mean(spec_bw) var(spec_bw) mean(rolloff) var(rolloff) mean(zcr) var(zcr) mean(harmony) var(harmony)'
for i in range(1, 21):
    header += f' mean(mfcc{i}) var(mfcc{i})'
header += ' label'
header = header.split()

#Dataset creation function
file = open('dataset.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
for g in genres:
    print(f'Genre:{g} started')
    start = time.clock()
    for filename in os.listdir(f'Data/genres_original/{g}'):
        songname = f'Data/genres_original/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=30,sr=None)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        length= librosa.get_duration(y=y, sr=sr) 
        tempo = float(librosa.beat.tempo(y=y, sr=sr)[0])
        harmony=librosa.feature.tonnetz(y=y, sr=sr)
        
        to_append = f'{filename} {length} {tempo} {np.mean(chroma_stft)} {np.var(chroma_stft)} {np.mean(rms)} {np.var(rms)} {np.mean(spec_cent)} {np.var(spec_cent)} {np.mean(spec_bw)} {np.var(spec_bw)} {np.mean(rolloff)} {np.var(rolloff)} {np.mean(zcr)} {np.var(zcr)} {np.mean(harmony)} {np.var(harmony)}'
        
        for e in mfcc:
            to_append += f' {np.mean(e)}'
            to_append += f' {np.var(e)}'
        to_append += f' {g}'
        file = open('datamain.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())
    print(f'Genre:{g} completed, took {time.clock()-start} seconds')
