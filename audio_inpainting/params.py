# -*- coding: utf-8 -*-
# Copyright (C) 2024 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import argparse
import time
import datetime
import os


def get_args():
    p = argparse.ArgumentParser(
        description="Audio inpainting with missing excerpts (masks) of 0.001-0.005 seconds"
    )
    p.add_argument('-ddc', '--clean-audio-file', default="./audio/target.wav", type=str) # path to clean/target audio file
    p.add_argument('-md', '--mask-duration', default=0.001, type=float) # duration of zero mask in seconds
    p.add_argument('-nm', '--num-masks', default=1, type=float) # number of zero masks
    p.add_argument('-ph', '--patch-height', default=1, type=int) # overlapping chunk height
    p.add_argument('-pw', '--patch-width', default=400, type=int) # overlapping chunk width, T
    p.add_argument('-k', '--Ksize', default=64, type=int) # number of variational parameters S
    p.add_argument('-ins', '--inner-net-shape', default=[512, 64]) # decoder: middle, hidden layer dimensions
    p.add_argument('-minlr', '--min-lr', default=0.0001, type=float) # minimum learning rate (cycling learning rate parameter)
    p.add_argument('-maxlr', '--max-lr', default=0.001, type=float) # maximum learning rate (cycling learning rate parameter)
    p.add_argument('-bs', '--batch-size', default=32, type=int) # batch size
    p.add_argument('-ne', '--no-epochs', default=501, type=int) # number of epochs
    p.add_argument('-ephc', '--epochs-per-half-cycle', default=20, type=int) # number of iterations in the increasing half of a cycle (cycling learning rate parameter)
    p.add_argument('-me', '--merge-every', default=20, type=int) # produce an audio reconstruction every (e.g., 20) epochs

    args = p.parse_args()

    args.output_directory = "./out/{}".format(
        os.environ["SLURM_JOBID"]
        if "SLURM_JOBID" in os.environ
        else datetime.datetime.fromtimestamp(time.time()).strftime("%y-%m-%d-%H-%M-%S")
    )

    return args
