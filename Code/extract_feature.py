import numpy as np
import os
import sys
from typing import Tuple
from tqdm import tqdm
from python_speech_features import *
import librosa
import librosa.display
from tensorflow.keras.utils import to_categorical
import argparse
from natsort import ns, natsorted
parser = argparse.ArgumentParser()

parser.add_argument('--mean_signal_length', type=int, default=96000)
parser.add_argument('--data_name', type=str, default="CASIA")
args = parser.parse_args()

def get_feature(file_path: str, feature_type:str="MFCC", mean_signal_length:int=96000, embed_len: int = 39):
    feature = None
    signal, fs = librosa.load(file_path)# Default setting on sampling rate
    s_len = len(signal)
    if s_len < mean_signal_length:
        pad_len = mean_signal_length - s_len
        pad_rem = pad_len % 2
        pad_len //= 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem), 'constant', constant_values = 0)
    else:
        pad_len = s_len - mean_signal_length
        pad_len //= 2
        signal = signal[pad_len:pad_len + mean_signal_length]
    if feature_type == "MFCC":
        mfcc =  librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=embed_len)
        feature = np.transpose(mfcc)
    return feature

def generate_csv(csv_save:str, data_name: str="EMODB", feature_type: str="MFCC", embed_len: int = 39, mean_signal_length:int = 96000, class_labels: Tuple = ("angry", "boredom", "disgust", "fear", "happy", "neutral","sad")):
    data_path = "./SER_WAV_DATA/"+data_name# Modify this path
    current_dir =  os.getcwd()
    if not os.path.exists(csv_save):
        print(csv_save+" build succeed")
        os.makedirs(csv_save)
        os.chdir(csv_save)
    else:
        os.chdir(csv_save)
    for i, directory in enumerate(class_labels):
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(directory+" build succeed")
    os.chdir('..')
    datapath = []
    labels = []
    sys.stderr.write('Current Folder: %s\n' % current_dir)
    os.chdir(data_path)
    for i, directory in enumerate(class_labels):
        sys.stderr.write("Start to Read %s\n" % directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
            os.chdir(directory)
            print(directory+" build succeed")
        else:
            os.chdir(directory)
        for filename in tqdm(os.listdir('.')):
            if not filename.endswith('wav'):
                continue
            filepath = os.getcwd() + '/' + filename
            datapath.append(filepath)
            labels.append(i)
        sys.stderr.write("End to Read %s\n" %directory)
        os.chdir('..')
    os.chdir(current_dir)
    temp_size_ = True
    for video_path,label in tqdm(zip(datapath,labels)):
        filename = video_path[video_path.rfind('/')+1:-4]
        feature_vector = get_feature(file_path = video_path, feature_type=feature_type, mean_signal_length=mean_signal_length, embed_len = embed_len)
        if temp_size_:
            print(f"### Feature Size:{feature_vector.shape} ###")
            temp_size_ = False
        np.savetxt(csv_save+"/" + class_labels[label] +"/" +filename+'_raw'+'.csv', feature_vector, delimiter = ',')


def process_csv(data_path: str, mfcc_len: int = 39, class_labels: Tuple = ("angry", "boredom", "disgust", "fear", "happy", "neutral","sad"), flatten: bool = False):
    x = []
    y = []
    current_dir =  os.getcwd()
    sys.stderr.write('Current Folder: %s\n' % current_dir)
    os.chdir(data_path)
    for i, directory in enumerate(class_labels):
        sys.stderr.write("Start to Read %s\n" % directory)
        os.chdir(directory)
        file_list = os.listdir('.')
        # file_list.sort(key=str.lower)# Sort by the file name
        file_list = natsorted(file_list,alg=ns.PATH)
        for filename in tqdm(file_list):
            if not filename.endswith('.csv'):
                continue
            if filename.endswith('time.csv'):
                continue
            filepath = os.getcwd() + '/' + filename
            feature_vector = np.loadtxt(filepath, delimiter=",", dtype = np.float32, encoding="gbk")
            x.append(feature_vector)
            y.append(i)
        sys.stderr.write("End to Read %s\n" %directory)
        os.chdir('..')
    os.chdir(current_dir)
    return np.array(x), np.array(y)

def extract_feature(data_name:str, feature_type_:str="MFCC", mean_signal_length:int=96000, class_labels:Tuple = ("angry", "boredom", "disgust", "fear", "happy", "neutral","sad")):
    csv_save = "./"+data_name+"_"+feature_type_+"_"+str(int(mean_signal_length/1000))
    generate_csv(csv_save=csv_save, data_name=data_name, class_labels=class_labels, feature_type=feature_type_, mean_signal_length=mean_signal_length)


EMODB_LABEL = ("angry", "boredom", "disgust", "fear", "happy", "neutral","sad")
CASIA_LABEL = ("angry", "fear", "happy", "neutral","sad","surprise")
SAVEE_LABEL =("angry","disgust", "fear", "happy", "neutral", "sad", "surprise")
RAVDE_LABEL = ("angry", "calm", "disgust", "fear", "happy", "neutral","sad","surprise")
IEMOCAP_LABEL = ("angry", "happy", "neutral", "sad")
EMOVO_LABEL = ("angry", "disgust", "fear", "happy","neutral","sad","surprise")
LABEL_DICT = {"CASIA":CASIA_LABEL,"EMODB":EMODB_LABEL,"IEMOCAP":IEMOCAP_LABEL,"EMOVO":EMOVO_LABEL,"SAVEE":SAVEE_LABEL,"RAVDE":RAVDE_LABEL}
PATH_DICT = {"CASIA":"./CASIA_MFCC_88","EMODB":"./EMODB_MFCC_96","IEMOCAP":"./IEMOCAP_MFCC_310","EMOVO":"./EMOVO_MFCC_96","SAVEE":"./SAVEE_MFCC_130","RAVDE":"./RAVDE_MFCC_110"}

# First step: extract speech feature
extract_feature(data_name=args.data_name, feature_type_="MFCC", mean_signal_length=args.mean_signal_length, class_labels=LABEL_DICT[args.data_name])
# Second step: convert .csv to .npy
x, y = process_csv(PATH_DICT[args.data_name], class_labels = LABEL_DICT[args.data_name], flatten = False)
y = to_categorical(y, num_classes=len(LABEL_DICT[args.data_name]))
data = {"x":x,"y":y}
np.save(args.data_name+".npy",data)
