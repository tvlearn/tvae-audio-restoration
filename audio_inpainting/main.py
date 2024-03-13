# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import os
import sys
import h5py
import numpy as np
import torch as to
import imageio

import tvo
from tvo.exp import EVOConfig, ExpConfig, Training
from tvo.models import GaussianTVAE

from tvutil.prepost import (
    OverlappingPatches
)

from params import get_args
from utils import stdout_logger, eval_fn, set_zero_mask, store_as_h5, eval_fn_wav

import soundfile as sf
import librosa

import time 

DEVICE = tvo.get_device()
PRECISION = to.float32
dtype_device_kwargs = {"dtype": PRECISION, "device": DEVICE}

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalized data
    source: https://stackoverflow.com/questions/49429734/trying-to-normalize-python-image-getting-error-rgb-values-must-be-in-the-0-1
    """
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))

def audio_inpainting():
    #start time
    start_time = time.time()

    # get hyperparameters
    args = get_args()
    print("Argument list:")
    for k in sorted(vars(args), key=lambda s: s.lower()):
        print("{: <25} : {}".format(k, vars(args)[k]))

    # determine directories to save output
    os.makedirs(args.output_directory, exist_ok=True)
    data_file, training_file = (
        args.output_directory + "/image_patches.h5",
        args.output_directory + "/training.h5",
    )
    txt_file = args.output_directory + "/terminal.txt"
    sys.stdout = stdout_logger(txt_file)  # type: ignore
    print("Will write training output to {}.".format(training_file))
    print("Will write terminal output to {}".format(txt_file))

    # generate incomplete image and extract image patches
    clean, sr = librosa.load(args.clean_audio_file, sr=16000) 
    clean = to.tensor(clean).to(**dtype_device_kwargs) 

    incomplete = set_zero_mask(clean, sr, args.mask_duration, args.num_masks)
    
    wav_file = f"{args.output_directory}/incomplete-{args.mask_duration}-missing.wav"
    incomplete_wav = incomplete.detach().cpu().numpy() 
    incomplete_for_save = np.nan_to_num(incomplete_wav, nan = 0.0)
    sf.write(wav_file, incomplete_for_save, sr)

    OVP = OverlappingPatches 
    ovp = OVP(incomplete[None, ...], args.patch_height, args.patch_width, patch_shift=1) 
    train_data = ovp.get().t()
    store_as_h5({"data": train_data}, data_file)

    D = args.patch_height * args.patch_width #* (3 if isrgb else 1)
    with h5py.File(data_file, "r") as f:
        N, D_read = f["data"].shape
        assert D == D_read

    print("Initializing model")

    model = GaussianTVAE(
        shape=[
            D,
        ]
        + args.inner_net_shape,
        min_lr=args.min_lr,
        max_lr=args.max_lr,
        cycliclr_step_size_up=np.ceil(N / args.batch_size) * args.epochs_per_half_cycle,
        precision=PRECISION,
    )

    print("Initializing experiment")

    # define hyperparameters of the variational optimization
    estep_conf = EVOConfig(
        n_states=args.Ksize,
        n_parents=5,
        n_children=4,
        n_generations=1,
        parent_selection="fitness",
        crossover=False,
    )

    # setup the experiment
    exp_config = ExpConfig(
        batch_size=32,
        output=training_file,
        reco_epochs=to.arange(args.no_epochs),
        log_blacklist=[
            "train_lpj",
            "train_states",
            "train_subs",
            "train_reconstruction",
        ],
        log_only_latest_theta=True,
    )
    exp = Training(
        conf=exp_config, estep_conf=estep_conf, model=model, train_data_file=data_file
    )
    logger, trainer = exp.logger, exp.trainer
    # append the noisy image to the data logger
    logger.set_and_write(incomplete_image=incomplete)

    # run epochs
    for epoch, summary in enumerate(exp.run(args.no_epochs)):
        summary.print()

        # merge reconstructed image patches and generate reconstructed image
        merge = ((epoch - 1) % args.merge_every) == 0
        assert hasattr(trainer, "train_reconstruction")
        reco = ovp.set_and_merge(trainer.train_reconstruction.t()) if merge else None

        # visualize epoch - images
        # TODO: SNR, PESQ here
        
        if merge:
            # TODO: combine eval_fn and eval_fn_wav!
            _, si_snr = eval_fn_wav(clean[None, ...], reco, DEVICE)
            psnr = eval_fn(clean[None, ...], reco) 
            to_log = {"reco_image": reco, "si-snr": si_snr, "psnr": psnr}
            if to_log is not None:
                logger.append_and_write(**to_log)
            si_snr_str = f"{si_snr:.2f}".replace(".", "_")
            psnr_str = f"{psnr:.2f}".replace(".", "_")
            wav_file = f"{args.output_directory}/reco-epoch{epoch-1}-si-snr{si_snr_str}-psnr{psnr_str}.wav"
            reco_audio = reco.squeeze(0).detach().cpu().numpy()
            
            sf.write(wav_file, reco_audio, sr)
            print(f"Wrote {wav_file}")

    print("Finished")
    end_time = time.time()
    print("Runtime: " + str(end_time-start_time) + ' seconds')

if __name__ == "__main__":
    audio_inpainting()
