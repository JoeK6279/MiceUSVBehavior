# Prediction of Mouse Behavior from USV Communication
This is the source code for the paper "A Deep Neural Network for Automatic Prediction of Mouse Behavior from USV Communication". 

The trained model weights can be downloaded [here](https://github.com/JoeK6279/MiceUSVBehavior/releases/download/v1.0/weights.zip).

## Installation
To use this code, first clone the repository and install necessary libraries.
```bash
git clone https://github.com/JoeK6279/MiceUSVBehavior
cd MiceUSVBehavior
pip install -U pip
pip install torch torchvision torchaudio # have to match with the cuda version
pip install -r requirements.txt
```

## Data
The dataset (including audio, video, and annotations) can be found at [here](https://osf.io/rvkb9/).

## Usage
First, generate the preprocessed audio dataset with:
```python
python preprocessing.py
```
### Aruguments
- `-d [duration]`: the target segment duartion in second
- `--datapath [path]`: audio files path
- `--outpath [path]`: targeted output path
The resulting segments would be stored under outpath.

Then to start training execute:
```python
python train.py
```
### Aruguments
- `-d [duration]`: the target segment duartion in second
- `--datapath [path]`: preprocessed dataset path
- `--bs [batch size]`: batch size for training
- `--TEST`: use when evaluation only

