

# Model info:
## please see README in folder "VGGnet model"


# audio capture - install instructions

## 1. prerequisites

    Pi4b with rasbian default OS and Respeaker mic installer.

### 1.1 check if you are you able to record audio from cli ?

    arecord -Dac108 -f S32_LE -r 16000 -c 4 hello.wav

    sox hello.wav -c 2 stereo.wav

    sudo apt-get install sox

    sox hello.wav -c 2 stereo.wav

    aplay stereo.wav

## 2. Lets install the Jupyter based capture app

#### 2.1 Commands from CLI

    sudo apt-get update

    sudo apt-get upgrade

    sudo pip3 install --upgrade pip

    sudo apt-get install ipython3

    sudo pip3 install jupyter

    sudo ipython3 kernelspec install-self

#### 2.2 clone the repo

    git clone https://github.com/sage7-ai/cough.git


#### 2.3 generate default config file for jupyter

    jupyter notebook --generate-config

#### 2.4 open the file:

    sudo nano ~/.jupyter/jupyter_notebook_config.py

#### 2.5 append the following lines at the bottom

    c.NotebookApp.allow_origin = '*' #allow all origins

    c.NotebookApp.ip = '0.0.0.0' # listen on all IPs

    c.NotebookApp.notebook_dir = '<replace with path to this git repo>'

#### 2.6 set jupyter as a service, create a file:
    sudo nano /etc/systemd/system/jupyter.service

#### 2.6.1 enter following and save the file and save it

    [Unit]

    After=network.service

    [Service]

    ExecStart=jupyter notebook

    ExecStop=jupyter notebook stop

    User= your-username-here

    [Install]

    WantedBy=default.target

#### 2.7 enable and start your service

    sudo systemctl enable jupyter.service

    sudo systemctl start jupyter.service

#### 2.8 restart the pi and jupyter should be running now

#### 2.9 run the commend and copy only the token part from it:

    jupyter notebook list

## 3.0 open from any laptop:

    http://<ip of the raspi>:8888

#### 3.1 enter the token copied earlier in the box, if you want you can set a password

#### 3.2 in case you dont see anything check ip and go back go back to pi cli and run this command to check status of jupyter;

    systemctl status  jupyter.service

#### 3.3 lets install widgets for notebook:

    sudo pip3 install ipywidgets notebook

    jupyter nbextension enable --py widgetsnbextension

    sudo apt install ffmpeg

    sudo apt-get install python3-pyaudio

    sudo pip3 install pocketsphinx webrtcvad

    sudo pip3 install respeaker --upgrade

    sudo pip3 install tqdm


## 4.0 optional installs

    pip3 install  webrtcvad mic_array pixel_ring

    pip3 install pyusb

    sudo apt-get install python-dev libatlas-base-dev swig

    pip3 install numpy

    sudo apt-get install python3-numpy

    pip3 install ipywebrtc

    pip3 install notebook
