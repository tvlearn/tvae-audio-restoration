# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import os
import sys
import h5py
import numpy as np
import torch as to
import matplotlib.pyplot as plt

import tvo
from tvo.exp import EVOConfig, ExpConfig, Training
from tvo.models import GaussianTVAE
from tvo.utils.param_init import init_sigma2_default

from tvutil.prepost import OverlappingPatches

from params import get_args 
from utils import stdout_logger, store_as_h5, eval_fn

import soundfile as sf
import librosa

import time 

DEVICE = tvo.get_device()
PRECISION = to.float32
dtype_device_kwargs = {"dtype": PRECISION, "device": DEVICE}

def audio_denoising():
    # start time
    start_time = time.time()
    # get hyperparameters
    args = get_args()
    print("Argument list:")
    for k in sorted(vars(args), key=lambda s: s.lower()):
        print("{: <25} : {}".format(k, vars(args)[k]))

    # determine directories to save output
    os.makedirs(args.output_directory, exist_ok=True)
    data_file, training_file = (
        args.output_directory + "/audio_chunks.h5",
        args.output_directory + "/training.h5",
    )
    txt_file = args.output_directory + "/terminal.txt"
    sys.stdout = stdout_logger(txt_file)  # type: ignore
    print("Will write training output to {}.".format(training_file))
    print("Will write terminal output to {}".format(txt_file))

    # read in the clean audio file at 16kHz
    clean, sr = librosa.load(args.clean_audio_file, sr=16000) 
    clean = to.tensor(clean).to(**dtype_device_kwargs) 

    # normalize if normalization is given 
    clean = clean / args.norm if args.norm is not None else clean

    # choose whether a noisy file should be used 
    if args.use_noisy:
        noisy, sr = librosa.load(args.noisy_audio_file, sr=16000)
        noisy = to.tensor(noisy).to(**dtype_device_kwargs)
    # or noise with sigma std is added to the clean signal and used as noisy
    else:
        sigma = args.sigma / args.norm if args.norm is not None else args.sigma
        noisy = clean + sigma * to.randn_like(clean) 

    # write noisy file to the output directory
    wav_file = f"{args.output_directory}/noisy-{args.sigma}-std.wav"
    noisy_wav = noisy.detach().cpu().numpy()
    sf.write(wav_file, noisy_wav, sr)

    # overlapping audio "chunks" are extracted (here called "patches")
    ovp = OverlappingPatches(noisy[None, ...], args.patch_height, args.patch_width, patch_shift=1)
    # overlapping chunks are the training data
    train_data = ovp.get().t()
    store_as_h5({"data": train_data}, data_file)

    # T: observed data dimension / chunk size
    T = args.patch_height * args.patch_width
    with h5py.File(data_file, "r") as f:
        data = to.tensor(f["data"][...])
    # N: number of training/inference datapoints
    N, T_read = data.shape
    assert T == T_read
    # initialize model parameter sigma
    sigma2_init = (
        (to.sqrt(init_sigma2_default(data, **dtype_device_kwargs)) / args.norm).pow(2)
        if args.norm is not None
        else init_sigma2_default(data, **dtype_device_kwargs)
    )
    del data

    print("Initializing model")

    # initialize model
    model = GaussianTVAE(
        shape=[
            T,
        ]
        + args.inner_net_shape,
        min_lr=args.min_lr,
        max_lr=args.max_lr,
        cycliclr_step_size_up=np.ceil(N / args.batch_size) * args.epochs_per_half_cycle,
        sigma2_init=sigma2_init,
        precision=PRECISION,
    )

    print("Initializing experiment")

    # define hyperparameters of the variational optimization
    estep_conf = EVOConfig(
        n_states=args.Ksize,
        n_parents=args.n_parents,
        n_children=args.n_children,
        n_generations=args.n_generations,
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
    # append the noisy audio to the data logger
    logger.set_and_write(noisy_image=noisy)

    # run epochs
    for epoch, summary in enumerate(exp.run(args.no_epochs)):
        summary.print()

        # merge reconstructed audio chunks and generate reconstructed audio
        merge = ((epoch - 1) % args.merge_every) == 0
        assert hasattr(trainer, "train_reconstruction")
        reco = ovp.set_and_merge(trainer.train_reconstruction.t()) if merge else None

        # save audio files if merge
        if merge:
            # calculate the objective metrics
            psnr, snr, pesq = eval_fn(clean[None, ...], reco) 
            to_log = {"reco_image": reco, "snr": snr, "pesq": pesq, "psnr": psnr}
            # add to data logger 
            if to_log is not None:
                logger.append_and_write(**to_log)
            snr_str = f"{snr:.2f}".replace("-", "m").replace(".", "_")
            pesq_str = f"{pesq:.2f}".replace(".", "_")
            psnr_str = f"{psnr:.2f}".replace(".", "_")
            wav_file = f"{args.output_directory}/reco-epoch{epoch-1}-snr{snr_str}-pesq{pesq_str}-psnr{psnr_str}.wav"
            reco_audio = reco.squeeze(0).detach().cpu().numpy()
            # write reconstruction audio file
            sf.write(wav_file, reco_audio, sr)
            print(f"Wrote {wav_file}")

    print("Finished")
    end_time = time.time()
    # calculate and print runtime
    print("Runtime: " + str(end_time-start_time) + ' seconds')


if __name__ == "__main__":
    audio_denoising()
