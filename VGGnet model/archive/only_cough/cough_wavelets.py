#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 2021

@author: Sudhir Vissa
"""

#Create melspoctrograms
import numpy as np
import pandas as pd
from scipy.io import wavfile
from sklearn.preprocessing import scale
import librosa.display
import librosa
import matplotlib.pyplot as plt
import os
import requests
import zipfile
import glob

import warnings
warnings.filterwarnings('ignore')
import scaleogram as scg
scg.set_default_wavelet('cmor1-1.5')


#%%

def download_url(url, save_path, chunk_size=128):

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    TOK=requests.post("https://ftp.basics.ai/api/login")
    headers = { "x-auth": TOK.text }
    r = requests.get(url, headers=headers, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)
    with zipfile.ZipFile(save_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(save_path))
    os.remove(save_path)

#%%

def save_wavelets_custom(directory_path, file_name, sampling_rate=44000):
    """ Will save spectogram into current directory"""
    
    path_to_file = os.path.join(directory_path, file_name)
    data, sr = librosa.load(path_to_file, sr=sampling_rate, mono=True)
    data = scale(data)

    melspec = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=128)
    log_melspec = librosa.power_to_db(melspec, ref=np.max)  
    librosa.display.specshow(log_melspec, sr=sr)
    
    # create saving directory
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    
    plt.savefig(directory_path + '/' + file_name.strip('.wav') + '.png', dpi = 300)
    os.remove(directory_path + '/' + file_name)

def save_wavelets_customnew(directory_path, file_name, sampling_rate=44000):
    """ Will save spectogram into current directory"""
    
    path_to_file = os.path.join(directory_path, file_name)
    data, sr = librosa.load(path_to_file, sr=sampling_rate, mono=True)
    data = scale(data)
    
    # trim silent edges
    n_fft = 2048
    hop_length=512
    n_mels=128
    whale_song, _ = librosa.effects.trim(data)

    S = librosa.feature.melspectrogram(whale_song, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length)


    # create saving directory
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    
    plt.savefig(directory_path + '/' + file_name.strip('.wav') + '.png', dpi = 300)
    os.remove(directory_path + '/' + file_name)

#%%

thepath = 'cough-dataset/'
download_url('https://ftp.basics.ai/api/raw/datasets/ECS50-dataset/coughing/coughs/?algo=zip', thepath+'honey.zip')
for filepath in glob.iglob(thepath+'*.wav'):
    print(filepath)
    save_wavelets_custom(os.path.dirname(filepath), os.path.basename(filepath), sampling_rate=44000)        


thepath = 'cough-samples/'
download_url('https://ftp.basics.ai/api/raw/cough/?algo=zip', thepath+'hoey.zip')
for filepath in glob.iglob(thepath+'*.wav'):
    print(filepath)
    save_wavelets_custom(os.path.dirname(filepath), os.path.basename(filepath), sampling_rate=44000)