#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 2021

@author: Sudhir Vissa
"""


#Create melspoctrograms


import numpy as np
import pandas as pd
import random
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

# data, sr = librosa.load('Datasets/ESC-50-master/audio/1-19111-A-24.wav', sr=5000, mono=True) # resplaced clock with cough
# plt.figure()
# scg_cwt = scg.cws(data, yscale='log')
# plt.savefig('cough_wavelet.png', dpi = 600)

#%%

# data, sr = librosa.load('Datasets/ESC-50-master/audio/1-19111-A-24.wav', sr=44100, mono=True) # resplaced clock with cough
# data = scale(data)
#%%
# plt.figure()
# plt.plot(data, linewidth=1, markersize=3)
# plt.savefig('cough_time.png', dpi = 600)
#%%

# plt.figure()
# melspec = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=128)
# log_melspec = librosa.power_to_db(melspec, ref=np.max)  
# librosa.display.specshow(log_melspec, sr=sr)
# plt.savefig('cough_spectrogram.png', dpi = 600)

#%%
# data, sr = librosa.load('Datasets/ESC-50-master/audio/1-30226-A-0.wav', sr=2000, mono=True)#replcaed alarm clock with dog

# scg_cwt = scg.cws(data, yscale='log')

#%%

# data, sr = librosa.load('Datasets/ESC-50-master/audio/1-13571-A-46.wav', sr=2000, mono=True) # replaced bells.wav

# scg_cwt = scg.cws(data, yscale='log')

#%%

# data, sr = librosa.load('Datasets/ESC-50-master/audio/1-26143-A-21.wav', sr=2000, mono=True) # replaced sneezing.

# scg_cwt = scg.cws(data, yscale='log')

# plt.savefig('sneeze_wt.png')
#%%


#%%

# def save_wavelets(directory_path, file_name, dataset_split, label, sampling_rate=5000):
#     """ Will save spectogram into current directory"""
    
#     path_to_file = os.path.join(directory_path, file_name)
#     data, sr = librosa.load(path_to_file, sr=sampling_rate, mono=True)
#     data = scale(data)

#     scg_cwt = scg.cws(data, yscale='log')
    
#     # create saving directory
#     directory = './wavelet_selected/{dataset}/{label}'.format(dataset=dataset_split, label=label)
#     if not os.path.exists(directory):
#         os.makedirs(directory)
    
#     plt.savefig(directory + '/' + file_name.strip('.wav') + '.png', dpi = 300)
    
    
# def _train_test_split(filenames, train_pct):
#     """Create train and test splits for ESC-50 data"""
#     random.seed(2018)
#     n_files = len(filenames)
#     n_train = int(n_files*train_pct)
#     train = np.random.choice(n_files, n_train, replace=False)
        
#     # split on training indices
#     training_idx = np.isin(range(n_files), train)
#     training_set = np.array(filenames)[training_idx]
#     testing_set = np.array(filenames)[~training_idx]
#     print('\tfiles in training set: {}, files in testing set: {}'.format(len(training_set), len(testing_set)))
    
#     return {'training': training_set, 'testing': testing_set}   




#%%

# dataset_dir = '../Datasets/ESC-50-master'

# # Load meta data for audio files
# meta_data = pd.read_csv(dataset_dir + '/meta/esc50.csv')

# labs = meta_data.category
# unique_labels = labs.unique()

# meta_data.head()

# unique_labels = np.array(['coughing', 'clapping', 'dog'])


# #%%

# for label in unique_labels:
    
#     print("Proccesing {} audio files".format(label))
    
#     current_label_meta_data = meta_data[meta_data.category == label]
    
#     datasets = _train_test_split(current_label_meta_data.filename, train_pct=0.8)
    
#     for dataset_split, audio_files in datasets.items():
        
#         for filename in audio_files:
            
#             directory_path = dataset_dir + '/audio/'
            
#             save_wavelets(directory_path, filename, dataset_split, label, sampling_rate=5000)
            
#%%
    # addding additional cough data sets:

def download_url(save_path, chunk_size=128):

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    url = 'https://ftp.basics.ai/api/raw/datasets/testing/?algo=zip'
    TOK=requests.post("https://ftp.basics.ai/api/login")
    headers = { "x-auth": TOK.text }
    r = requests.get(url, headers=headers, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)
    with zipfile.ZipFile(save_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(save_path))
    os.remove(save_path)


def save_wavelets_custom(directory_path, file_name, sampling_rate=5000):
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


thepath = 'wavelet_testing/'
download_url(thepath+'any.zip')

for filepath in glob.iglob(thepath+'*/'+'*.wav'):
    print(filepath)
    save_wavelets_custom(os.path.dirname(filepath), os.path.basename(filepath), sampling_rate=5000)