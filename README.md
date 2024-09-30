# Blind Zero-Shot Audio Restoration: A Variational Autoencoder Approach for Denoising and Inpainting

## Overview 
The [audio denoising](./audio_denoising) and [audio inpainting](./audio_inpainting) directories contain implementations of the experiments described in the [paper](https://www.isca-archive.org/interspeech_2024/boukun24_interspeech.html). Execution requires an installation of the [Truncated Variational Optimization](https://github.com/tvlearn/tvo) (TVO) framework, which implements the Truncated Variational Autoencoder (TVAE). Experiments furthermore leverage pre-/postprocessing provided by [tvutil](https://github.com/tvlearn/tvutil).

Please follow the [Setup](#setup) instructions described below to run the experiments. 

The code has only been tested on Linux systems.

## Setup
We recommend [Anaconda](https://www.anaconda.com/) to manage the installation, and to create a new environment for hosting the installed packages:

```bash
$ conda env create -n tvae-audio python==3.8.10 gcc_linux-64 
$ conda activate tvae-audio
```
Mac users can comment out ```gcc_linux-64```. 
Install the required packages via pip:

```bash
$ pip install -r requirements.txt
```

The `tvo` package can be installed via:

```bash
$ git clone -b tvae-audio-restoration https://github.com/tvlearn/tvo.git
$ cd tvo
$ python setup.py build_ext
$ python setup.py install
$ cd ..
```

To install `tvutil`, run:

```bash
$ git clone https://github.com/tvlearn/tvutil.git
$ cd tvutil
$ python setup.py install
$ cd ..
```
To perform audio denoising or audio inpainting, add the corrupted and, if available, the clean audio files (.wav) to the directories audio_denoising/audio/ or audio_inpaining/audio/, correspondingly.

To train TVAE on a corrupted audio file run:
```bash
$ env HDF5_USE_FILE_LOCKING='FALSE' python main.py  
```
Please check params.py file to see which parameters can be modified (such as paths to your corrupted files, whether only noisy file is provided, etc.) 

To exploit GPU parallelization, run ```env HDF5_USE_FILE_LOCKING='FALSE' TVO_GPU=0 python main.py```. GPU execution requires a cudatoolkit installation. 

## Reference

```bibtex
@inproceedings{
    title={{Blind Zero-Shot Audio Restoration: A Variational Autoencoder Approach for Denoising and Inpainting. Proc. Interspeech}},
    author={Boukun, Veranika and Drefs, Jakob and L{\"u}cke, J{\"o}rg},
    booktitle = {Proc. Interspeech 2024},
    year={2024},
    pages = {4823--4827},
}
```
