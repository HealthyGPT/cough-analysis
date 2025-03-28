# Cough Edge data acquisition model with transfer learning

## High level details
1. Data collection is from edge device for testing feasibility in listetning to the patients.

    For a comprehensive list of tests please see the jupyter notebook "Tests.ipynb" on data collection procedure

    Data collection happens at 44 kHz and is recorded on the edge device ("Tests.ipynb" is run on the edge device)

    once tests are complete all test data ("Wave files") is pushed into the "ftp file server".

2. Model can be trained or run from any device/server (not tested on edge device)

    Training model: currently we are training the model to recognize clapping, coughing and dog sounds

    Model training takes ~20 mins on a local machine with 16GB ram.

    Model training takes sample set of 40 wav files for each label and divides them 80% for trianing and 20% for testing.(32 for training/label and 8 for testing/label)

    Model training data is acquired from EC-50 open source dataset.

3. Rendering model: currently this is tested on servers/remote machines (devices other than edge)

    all live data collected from #1 is pulled down from "ftp file server" 

    Data is pre rpocessed from ".wav" file to generated mel-spectrograms into ".png" using librosa library 
    
    During preprocessing, we are using a sampling rate of 5kHz rate(high rate takes longer time to draw spectrograms, 44khz takes longer to carve out mel-spectrograms with little improvement in the model)


## 3 different models have been tried and tested on the collected dataset from raspberry pi.

## The VGGNet-16 Model
     Cough detection model runs with Log Mel Spectrogram, Wavelet Transformation, Deep learning and Transfer learning concepts.
    •	A computer vision and deep learning (with transfer learning) framework for detection of cough sounds
    •	Log Mel spectrogram and Wavelet transform images were obtained for sound data samples (ESC-50)
    •	A CNN (Convolutional Neural Networks) based pretrained model "VGG16" with a thin topmodel wrapper was trained on this image dataset to detect cough sounds.
    * results from this model identify COugh and Dog intermittently.
    
## Cough detection using YAMNet Model
     YAMNet is a pre-trained deep neural network that can predict audio events from 521 classes, such as coughing, laughter, barking, or a siren.
     In this model, YAMNet is used for high level feature extraction - the 1024 dimensional embedding output. This is similar to transfer learning for image classification
     For training the model, ESC-50 filtered dataset is used. The filtered dataset has # Clapping, Dog and Coughing classes only. Since we are using the transfer learning we do not need to have lots of labeled training data. About the dataset - https://www.karolpiczak.com/papers/Piczak2015-ESC-Dataset.pdf
     Here is the model layer information
     ![Transfer_learning_audio_model_plot](https://www.tensorflow.org/tutorials/audio/transfer_learning_audio_files/output_y-0bY5FMme1C_0.png)

     * Model interanlly uses SparseCategoricalCrossentropy. For info please visit: https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy
     
## Simple Linear Audio Classification Model

     This model uses simple convolutional neural network (CNN).
     We transform the audiofiles to spectrograms and give those images as input to the CNN model.
     This model uses filtered ESC-50 dataset similar to YAMNet model for training purposes. In preprocessing decoded tensors were created for the waveforms and the corresponding labels.
     Here is the model info
 
     Model: "sequential"
    _________________________________________________________________

     Layer (type)                   Output Shape              Param #   
    _________________________________________________________________

     resizing (Resizing)           (None, 32, 32, 1)         0         
                                                                     
     normalization (Normalization) (None, 32, 32, 1)         3         
                                                                     
     conv2d (Conv2D)               (None, 30, 30, 32)        320       
                                                                     
     conv2d_1 (Conv2D)             (None, 28, 28, 64)        18496     
                                                                     
     max_pooling2d (MaxPooling2D)  (None, 14, 14, 64)        0                                                                     
                                                                     
     dropout (Dropout)             (None, 14, 14, 64)        0         
                                                                     
     flatten (Flatten)             (None, 12544)             0         
                                                                     
     dense (Dense)                 (None, 128)               1605760   
                                                                     
     dropout_1 (Dropout)           (None, 128)               0         
                                                                     
     dense_1 (Dense)               (None, 3)                 387       
                                                                     
    _________________________________________________________________
    Total params: 1,624,966
    Trainable params: 1,624,963
    Non-trainable params: 3
    ________________________________________________________________

    This model uses Cross Entropy loss and Adam optimizer.




## References
    More details about the ESC-50 data: https://github.com/karolpiczak/ESC-50
    Additional cough datasets used: https://zenodo.org/record/5136592#.YdR0thNKirz
    More details about VGG16 model - https://neurohive.io/en/popular-networks/vgg16/


# Notes
    **The accuracy of models greatly depend on quality of audio files and the features available within the audio. Single clap vs Double clap would make lot of difference.
    ***Clapping cannot be identified due to lack in dataasets on singular clap seq
    *Edit alpha 0.5 from '0.5' in /usr/local/lib/python3.9/site-packages/scaleogram/cws.py library.
