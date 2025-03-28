# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 22:54:57 2022

@author: Ameer
"""


import os
import pathlib
from IPython import display
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
import sklearn
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


DATASET_PATH = 'D:/Vissa/cough-detection-with-transfer-learning/Datasets/ESC-50-master/audio'
data_dir = pathlib.Path(DATASET_PATH)

filenames = tf.io.gfile.glob(str(data_dir) + '/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)
print('Number of total examples:', num_samples)
print('Example file tensor:', filenames[0])

esc50_csv = 'D:/Vissa/cough-detection-with-transfer-learning/Datasets/ESC-50-master/meta/esc50.csv'
base_data_path = 'D:/Vissa/cough-detection-with-transfer-learning/Datasets/ESC-50-master/audio/'

pd_data = pd.read_csv(esc50_csv)
pd_data.head()

my_classes = ['clapping', 'dog', 'coughing']
map_class_to_id = {'clapping':0, 'dog':1, 'coughing':2}

filtered_pd = pd_data[pd_data.category.isin(my_classes)]
### Use iterative stratifying for train test split.

class_id = filtered_pd['category'].apply(lambda name: map_class_to_id[name])
filtered_pd = filtered_pd.assign(target=class_id)

full_path = filtered_pd['filename'].apply(lambda row: os.path.join(base_data_path, row))
filtered_pd = filtered_pd.assign(filename=full_path)

filtered_pd.head(5)
filtered_pd.shape


filenames = filtered_pd['filename']
targets = filtered_pd['target']

filenames = filenames.reset_index()
targets = targets.reset_index()

filenames.shape      # (120,2)
targets.shape        # (120,2)

## CONVERT TO np.array
filenames =  np.array(filenames)
targets = np.array(targets)

from skmultilearn.model_selection import iterative_train_test_split
filenames_train, targets_train, filenames_test, targets_test = iterative_train_test_split(filenames, targets, test_size = 0.2)
filenames_train = pd.DataFrame(filenames_train[1:, 1:], columns=['filepath'])
filenames_test = pd.DataFrame(filenames_test[1:, 1:], columns=['filepath'])
targets_train = pd.DataFrame(targets_train[1:, 1:], columns=['category'])
targets_test = pd.DataFrame(targets_test[1:, 1:], columns=['category'])


#train_filenames = 





""" Preprocess wav file to audio tensors """

def decode_audio(audio_binary):
  # Decode WAV-encoded audio files to `float32` tensors, normalized
  # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
  audio, _ = tf.audio.decode_wav(contents=audio_binary)
  # Since all the data is single channel (mono), drop the `channels`
  # axis from the array.
  return tf.squeeze(audio, axis=-1)

"""

Split the file paths into tf.RaggedTensors (tensors with ragged dimensions—with slices that may have different lengths).

"""

### NOT NEEDED TO RUN


def get_label(file_path):
  parts = tf.strings.split(
      input=file_path,
      sep=os.path.sep)
  # Note: You'll use indexing here instead of tuple unpacking to enable this
  # to work in a TensorFlow graph.
  return parts[-2]



'''
Define another helper function—get_waveform_and_label—that puts it all together:

The input is the WAV audio filename.
The output is a tuple containing the audio and label tensors ready for supervised learning.
'''

def get_waveform_and_label(file_path, label):
  #label = get_label(file_path)
  label = label
  print("label = ", label)
  print("file_path", file_path)
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label

"""
 Build the training set to extract audio-label pairs
 
"""



AUTOTUNE = tf.data.AUTOTUNE

filenames_train = np.array(filenames_train)
targets_train = np.array(targets_train)

filenames_test = np.array(filenames_test)
targets_test =  np.array(targets_test)


train_ds = tf.data.Dataset.from_tensor_slices((filenames_train, targets_train))
test_ds = tf.data.Dataset.from_tensor_slices((filenames_test, targets_test))

train_ds = train_ds.unbatch()

waveform_train_ds = train_ds.map(get_waveform_and_label,
    num_parallel_calls=AUTOTUNE)

#waveform_train_ds = waveform_train_ds.unbatch()


''' 
Convert waveforms to spectrograms

'''

def get_spectrogram(waveform):
  # Zero-padding for an audio waveform with less than 16,000 samples.
  input_len = 16000
  waveform = waveform[:input_len]
  zero_padding = tf.zeros(
      [16000] - tf.shape(waveform),
      dtype=tf.float32)
  # Cast the waveform tensors' dtype to float32.
  waveform = tf.cast(waveform, dtype=tf.float32)
  # Concatenate the waveform with `zero_padding`, which ensures all audio
  # clips are of the same length.
  equal_length = tf.concat([waveform, zero_padding], 0)
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

for waveform, label in waveform_train_ds.take(1):
  label = label
  spectrogram = get_spectrogram(waveform)

print('Label:', label)
print('Waveform shape:', waveform.shape)
print('Spectrogram shape:', spectrogram.shape)
print('Audio playback')
display.display(display.Audio(waveform, rate=16000))

def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)
  
fig, axes = plt.subplots(2, figsize=(12, 8))
timescale = np.arange(waveform.shape[0])
axes[0].plot(timescale, waveform.numpy())
axes[0].set_title('Waveform')
axes[0].set_xlim([0, 16000])

plot_spectrogram(spectrogram.numpy(), axes[1])
axes[1].set_title('Spectrogram')
plt.show()

"""     Transform waveform dataset to Spectogram 
        and their corresponding labels as integers
"""


def get_spectrogram_and_label_id(audio, label):
  spectrogram = get_spectrogram(audio)
  label_id = tf.argmax(label == [0, 1, 2])
  return spectrogram, label_id

'''Map get_spectrogram_and_label_id across the dataset's elements with Dataset.map:

'''


spectrogram_ds = waveform_train_ds.map(
  map_func=get_spectrogram_and_label_id,
  num_parallel_calls=AUTOTUNE)



"""Build and train the model
"""

def preprocess_dataset(files):
  #files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files.map(
      map_func=get_waveform_and_label,
      num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(
      map_func=get_spectrogram_and_label_id,
      num_parallel_calls=AUTOTUNE)
  return output_ds

train_ds = spectrogram_ds
#val_ds = preprocess_dataset(val_files)
test_ds = test_ds.unbatch()
test_ds = preprocess_dataset(test_ds)


batch_size = 64
train_ds = train_ds.batch(batch_size)
#val_ds = val_ds.batch(batch_size)
test_ds = test_ds.batch(batch_size)

## CACHE TO REDUCE THE LATENCY IN READING FILES

train_ds = train_ds.cache().prefetch(AUTOTUNE)
#val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)
####https://www.tensorflow.org/tutorials/audio/simple_audio

for spectrogram, _ in spectrogram_ds.take(1):
   print(spectrogram.shape)
   input_shape = spectrogram.shape
print('Input shape:', input_shape)
num_labels = len(my_classes)

# Instantiate the `tf.keras.layers.Normalization` layer.
norm_layer = layers.Normalization()
# Fit the state of the layer to the spectrograms
# with `Normalization.adapt`.
norm_layer.adapt(data=spectrogram_ds.map(map_func=lambda spec, label: spec))

model = models.Sequential([
    layers.Input(shape=input_shape),
    # Downsample the input.
    layers.Resizing(32, 32),
    # Normalize.
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels),
])

model.summary()


model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)


EPOCHS = 20
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=8),
)


metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()


""" EVALUATE THE MODEL PERFORMANCE """

test_audio = []
test_labels = []

for audio, label in test_ds:
  test_audio.append(audio.numpy())
  test_labels.append(label.numpy())

test_audio = np.array(test_audio)
test_labels = np.array(test_labels)

y_pred = np.argmax(model.predict(test_audio), axis=1)
y_true = test_labels

test_acc = sum(y_pred == y_true) / len(y_true)
print(f'Test set accuracy: {test_acc:.0%}')

"""Display a confusion matrix"""
confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx,
            xticklabels=my_classes,
            yticklabels=my_classes,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

"""Run inference on an audio file"""



def get_waveform_and_label_inf(file_path):
  label = get_label(file_path)
  #label = label
  print("label = ", label)
  print("file_path", file_path)
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label

def get_spectrogram_and_label_id_inf(audio, label):
  spectrogram = get_spectrogram(audio)
  label_id = tf.argmax(label == my_classes)
  print("label_id  is: ", label_id)
  return spectrogram, label_id

def preprocess_dataset_inf(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(
      map_func=get_waveform_and_label_inf,
      num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(
      map_func=get_spectrogram_and_label_id_inf,
      num_parallel_calls=AUTOTUNE)
  return output_ds

sample_file = 'D:\\Vissa\\cough-detection-with-transfer-learning\\new_predict_sounds\\rob_1.1.1.wav'
#sample_file= tf.io.read_file(sample_file)

sample_file = tf.data.Dataset.from_tensor_slices([str(sample_file)])

sample_ds = preprocess_dataset_inf([str(sample_file)])

for spectrogram, label in sample_ds.batch(1):
  prediction = model(spectrogram)
  plt.bar(my_classes, tf.nn.softmax(prediction[0]))
  plt.title(f'Predictions for "{my_classes[label[0]]}"')
  plt.show()

