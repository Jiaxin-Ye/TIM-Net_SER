# TIM-Net

Official Tensorflow implementation of ICASSP 2023 paper, "Temporal Modeling Matters: A Novel Temporal Emotional Modeling Approach for Speech Emotion Recognition". [[paper]](https://arxiv.org/abs/2211.08233) [[code]](https://github.com/Jiaxin-Ye/TIM-Net_SER) 

### Introduction

In this paper, we propose a **T**emporal-aware b**I**-direction **M**ulti-scale Network, termed **TIM-Net**, which is a novel temporal emotional modeling approach to learn multi-scale contextual affective representations from various time scales. 

![architecture](./Fig/architecture.png)

## Usage:

> **Our MFCC features files (*.npy)**: 
> 
> `Baidu links`: https://pan.baidu.com/s/1Y-GDJXpF0FqjcGGN6y84JA?pwd=MFCC `code`: MFCC 
> 
> `Google links`: https://drive.google.com/drive/folders/1nl7hej--Nds2m3MrMDHT63fNL-yRRe3d
> 
>
>**Our testing model weight files (*.hdf5)**: 
> 
> `Baidu links`:  https://pan.baidu.com/s/1EtjhuljeHwvIjYG8hYtMXQ?pwd=HDF5 `code`: HDF5
> 
> `Google links`: https://drive.google.com/drive/folders/1ZjgcT6R0A0C2twXatgpimFh1a3IL81pw

### 1. Clone Repository

```bash
$ git clone https://github.com/Jiaxin-Ye/TIM-Net_SER.git
$ cd TIM-Net_SER/Code
```

### 2. Requirements

Our code is based on Python 3 (>= 3.8). There are a few dependencies to run the code. The major libraries are listed as follows:

* Tensorflow-gpu (>= 2.5.0)
* Keras (>= 2.5.0, the same as TF)
* Scikit-learn (== 1.0.2)
* NumPy (>= 1.19.5)
* SciPy (== 1.8.0)
* librosa (== 0.8.1)
* Pandas (== 1.4.1)
* ......

```bash
$ pip install -r requirement.txt
```

### 3. Datasets

The six public emotion datasets are used in the experiments: the CASIA, EMODB, EMOVO, IEMOCAP, RAVDESS, and SAVEE. The languages of IEMOCAP, RAVDESS and SAVEE are English, while the CASIA, EMODB and EMOVO datasets contain Chinese, German and Italian speeches.

### 4. Preprocessing

In the experiments, the 39-D static MFCCs are extracted using the Librosa toolbox with the **default settings**. Specifically, the frame length is 50 ms, the frame shift is 12.5 ms, the sample rate is 22050 Hz and the window function added for the speech is Hamming window. 

> In the single-corpus SER task, the "mean_signal_length" is set to 88000, 96000, 96000, 310000, 110000, 130000 for CASIA, EMODB, EMOVO, IEMOCAP, RAVDESS and SAVEE, which is almost the same as the maximum length of the input sequences.

If you are not convenient  to preprocess these features, you can download them from our shared [link](https://pan.baidu.com/s/1Y-GDJXpF0FqjcGGN6y84JA?pwd=MFCC ) to `MFCC` folder.

```python
def get_feature(file_path: str, mfcc_len: int = 39, mean_signal_length: int = 100000):
  	"""
    file_path: Speech signal folder
    mfcc_len: MFCC coefficient length
    mean_signal_length: MFCC feature average length
  	"""
    signal, fs = librosa.load(file_path)
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
    mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=39)
    mfcc = mfcc.T
    feature = mfcc
    return feature
```

### 5. Training and Testing

#### Training

```bash
$ python main.py --mode train --data EMODB --split_fold 10 --random_seed 46 --epoch 300 --gpu 0
```

#### Testing

If you want to test your model on 10-fold cross-validation manner with `X' random seed, you can run following commands:

```bash
$ python main.py --mode test --data CASIA  --test_path ./Test_Models/CASIA_32 --split_fold 10 --random_seed 32
$ python main.py --mode test --data EMODB  --test_path ./Test_Models/EMODB_46 --split_fold 10 --random_seed 46
$ python main.py --mode test --data EMOVO  --test_path ./Test_Models/EMOVO_1 --split_fold 10 --random_seed 1
$ python main.py --mode test --data IEMOCAP  --test_path ./Test_Models/IEMOCAP_16 --split_fold 10 --random_seed 16
$ python main.py --mode test --data RAVDE  --test_path ./Test_Models/RAVDE_46 --split_fold 10 --random_seed 46
$ python main.py --mode test --data SAVEE  --test_path ./Test_Models/SAVEE_44 --split_fold 10 --random_seed 44
```

You can download our model files from our shared [link]( https://pan.baidu.com/s/1EtjhuljeHwvIjYG8hYtMXQ?pwd=HDF5) to `Test_Models` folder. 

### 6. Training Details

#### 6.1 Experiment Settings

The **cross-entropy criterion** is used as the objective function. Adam algorithm is adopted to optimize the model with an initial **learning rate** $\alpha$ = $0.001$, and a **batch size** of 64. To avoid over-fitting during the training phase, we implement label **smoothing with factor** 0.1 as a form of regularization and set the **spatial dropout rate** to 0.1. For the $j$-th TAB $\mathcal{T}_j$, there are 39 **kernels** of **size** 2 in Conv layers, and the **dilated rate** is $2^{j-1}$. To guarantee that the maximal receptive field covers the input sequences, we set the **number of TAB** $n$ in both directions to 10 for IEMOCAP and 8 for others.

For fair comparisons with the SOTA approaches, we perform **10-fold cross-validation (CV)** as well as previous works in experiments. 

#### 6.2 Model Size

Since not all SOTA methods we compared provide their source codes or model sizes in the paper, we can only select some typical ones for size comparison. For example, the model sizes of *Light-SERNet* (0.88 MB), *GM-TCN* (1.13 MB), and *CPAC* (1.23 MB) are all larger than TIM-Net (**0.40** MB). 

Our proposed models are trained on an Nvidia GeForce RTX 3090 GPU with an average of 60 ms per step. The results demonstrate that our TIM-Net is **lightweight** yet **effective**.

#### 6.3 Training Procedure

We show the training procedure on the EMODB dataset in the following figure, which illustrates that our model can converge effectively.

![architecture](./Fig/training.png)

## Folder structure

```
TIM-NEt_SER
├─ Code
│    ├─ MFCC (Download MFCC files here)
│    ├─ Models (Store model files)
│    ├─ Results (Store result files)
│    ├─ Test_Models (Download pretrained models here)
│    ├─ Model.py
│    ├─ TIMNET.py
│    ├─ Common_Model.py
│    ├─ main.py
│    └─ requirement.txt
├─ README.md
```

## Acknowledgement

- We thank the reviewers for the insightful feedback and constructive suggestions.
- If you compare this paper or use our code in your research, please consider citing us as follows.
```bibtex
@inproceedings{TIMNET,
  title={Temporal Modeling Matters: A Novel Temporal Emotional Modeling Approach for Speech Emotion Recognition},
  author = {Ye, Jiaxin and Wen, Xincheng and Wei, Yujie and Xu, Yong and Liu, Kunhong and Shan, Hongming},
  booktitle = {ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Rhodes Island, Greece, June 4-10, 2023},
  pages={1--5},
  year = {2023}
}
```
- If you are interested in speech emotion recognition, we are delighted to recommend our latest SER works for you to follow up: 
  
  [CTL-MTNet](https://www.ijcai.org/proceedings/2022/0320.pdf): A Novel CapsNet and Transfer Learning-Based Mixed Task Net for the Single-Corpus and Cross-Corpus Speech Emotion Recognition, _**IJCAI 2022**_.
  
  [GM-TCNet](https://arxiv.org/abs/2210.15834): Gated Multi-scale Temporal Convolutional Network using Emotion Causality for Speech Emotion Recognition, _**Speech Communication 2022**_.
  


