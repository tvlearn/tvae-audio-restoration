# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import argparse
import time
import datetime
import os


def get_args():
    p = argparse.ArgumentParser(
        description="Audio inpainting with missing excerpts (masks) of 0.1-0.25 seconds"
    )
    p.add_argument('-ddc', '--clean-audio-file', default=".../target.wav", type=str)
    p.add_argument('-md', '--mask-duration', default=0.002, type=float)
    p.add_argument('-nm', '--num-masks', default=1, type=float)
    p.add_argument('-p', '--percentage', default=80, type=int)
    p.add_argument('-ph', '--patch-height', default=1, type=int)
    p.add_argument('-pw', '--patch-width', default=400, type=int)
    p.add_argument('-k', '--Ksize', default=64, type=int)
    p.add_argument('-ins', '--inner-net-shape', default=[512, 64]) 
    p.add_argument('-minlr', '--min-lr', default=0.0001, type=float) 
    p.add_argument('-maxlr', '--max-lr', default=0.001, type=float)
    p.add_argument('-bs', '--batch-size', default=32, type=int)
    p.add_argument('-ne', '--no-epochs', default=501, type=int)
    p.add_argument('-ephc', '--epochs-per-half-cycle', default=20, type=int)
    p.add_argument('-me', '--merge-every', default=20, type=int)

    args = p.parse_args()

    args.output_directory = ".../out/{}".format(
        os.environ["SLURM_JOBID"]
        if "SLURM_JOBID" in os.environ
        else datetime.datetime.fromtimestamp(time.time()).strftime("%y-%m-%d-%H-%M-%S")
    )

    return args
