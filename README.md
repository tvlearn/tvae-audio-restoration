# Blind Zero-Shot Audio Restoration: A Variational Autoencoder Approach for Denoising and Inpainting

## Overview 
The [audio denoising](./audio_denoising) and [audio inpainting](./audio_inpainting) directories contain implementations of the experiments described in the [paper](). Execution requires an installation of the [Truncated Variational Optimization](https://github.com/tvlearn/tvo) (TVO) framework, which implements the Truncated Variational Autoencoder (TVAE). Experiments furthermore leverage pre-/postprocessing provided by [tvutil](https://github.com/tvlearn/tvutil).

Please follow the [Setup](#setup) instructions described below to run the experiments. Please consult the README files in the sub-directories for further instructions.

The code has only been tested on Linux systems.

## Setup
We recommend [Anaconda](https://www.anaconda.com/) to manage the installation, and to create a new environment for hosting the installed packages:

```bash
$ conda env create -n tvae-audio python==3.8.10 gcc_linux-64 
$ conda activate tvae-audio
```
Mac users can comment out ```bash gcc_linux-64```. 
Install the required packages via pip:

```bash
$ pip install -r requirements.txt
```

The `tvo` package can be installed via:

```bash
$ git clone https://github.com/tvlearn/tvo.git
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

## Reference

```bibtex
@inproceedings{}
```
