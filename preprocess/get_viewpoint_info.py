#!/usr/bin/env python3

''' Script to precompute image features using a Pytorch ResNet CNN, using 36 discretized views
    at each viewpoint in 30 degree increments, and the provided camera WIDTH, HEIGHT 
    and VFOV parameters. '''

import os
import sys
import logging
import Matterport3DSimulator.MatterSim as MatterSim
import math
from tqdm import tqdm
import cv2
import argparse
import numpy as np
import json
import math
import h5py
import copy
from PIL import Image
import time
from progressbar import ProgressBar

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from utils import load_viewpoint_ids
import os
import os.path as op



TSV_FIELDNAMES = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features']
VIEWPOINT_SIZE = 36 # Number of discretized views from one viewpoint

WIDTH = 128
HEIGHT = 128
VFOV = 60
viewpoint_info_dict = {}

def build_simulator(connectivity_dir, scan_dir):
    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setDatasetPath(scan_dir)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setDepthEnabled(True)
    sim.setPreloadingEnabled(False)
    sim.setBatchSize(1)
    sim.initialize()
    return sim




def build_feature_file(args):

    scanvp_list = load_viewpoint_ids(args.connectivity_dir)

    # Set up the simulator
    sim = build_simulator(args.connectivity_dir, args.scan_dir)


    for scan_id, viewpoint_id in tqdm(scanvp_list):
        # Loop all discretized views from this location
        sim.newEpisode([scan_id], [viewpoint_id], [0], [0])  #['17DRP5sb8fy'] ['10c252c90fa24ef3b698c6f54d984c5c']
        state = sim.getState()[0]
        key = '%s_%s' % (scan_id, viewpoint_id)
        viewpoint_info_dict[key]={"x":state.location.x,"y":state.location.y,"z":state.location.z}

    json.dump(viewpoint_info_dict,open(args.output_file,"w"))
           
            

            
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--connectivity_dir', default='connectivity')
    parser.add_argument('--scan_dir', default='../data/v1/scans')
    parser.add_argument('--output_file',)

    args = parser.parse_args()

    build_feature_file(args)
    #read_features(args)


