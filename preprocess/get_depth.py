#!/usr/bin/env python3

''' Script to precompute depth using 36 discretized views
    at each viewpoint in 30 degree increments, and the provided camera WIDTH, HEIGHT 
    and VFOV parameters. '''

import os
import sys
import logging
import MatterSim
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

def process_features(proc_id, out_queue, scanvp_list, args):
    print('start proc_id: %d' % proc_id)

    # Set up the simulator
    sim = build_simulator(args.connectivity_dir, args.scan_dir)


    for scan_id, viewpoint_id in scanvp_list:
        # Loop all discretized views from this location
        images = []
        for ix in range(VIEWPOINT_SIZE):
            
            if ix == 0:
                sim.newEpisode([scan_id], [viewpoint_id], [0], [math.radians(-30)])  #['17DRP5sb8fy'] ['10c252c90fa24ef3b698c6f54d984c5c']
            elif ix % 12 == 0:
                sim.makeAction([0], [1.0], [1.0])
            else:
                sim.makeAction([0], [1.0], [0])
            state = sim.getState()[0]
            assert state.viewIndex == ix

            if 12 <= ix and ix < 24:
                pass
            else:
                continue

            image = np.array(state.depth, copy=True).reshape(1,WIDTH,HEIGHT,1) 

            images.append(image)
        images = np.concatenate(images,axis=0)
        out_queue.put((scan_id, viewpoint_id, images))
        

    out_queue.put(None)


def build_feature_file(args):
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    scanvp_list = load_viewpoint_ids(args.connectivity_dir)

    num_workers = min(args.num_workers, len(scanvp_list))
    num_data_per_worker = len(scanvp_list) // num_workers

    out_queue = mp.Queue()
    processes = []
    for proc_id in range(num_workers):
        sidx = proc_id * num_data_per_worker
        eidx = None if proc_id == num_workers - 1 else sidx + num_data_per_worker

        process = mp.Process(
            target=process_features,
            args=(proc_id, out_queue, scanvp_list[sidx: eidx], args)
        )
        process.start()
        processes.append(process)
    
    num_finished_workers = 0
    num_finished_vps = 0

    progress_bar = ProgressBar(max_value=len(scanvp_list))
    progress_bar.start()

    with h5py.File(args.output_file, 'w') as outf:
        while num_finished_workers < num_workers:
            res = out_queue.get()
            if res is None:
                num_finished_workers += 1
            else:
                scan_id, viewpoint_id, fts = res
                key = '%s_%s'%(scan_id, viewpoint_id)

                data = fts
                outf.create_dataset(key, data.shape, dtype='float', compression='gzip')
                outf[key][...] = data
                outf[key].attrs['scanId'] = scan_id
                outf[key].attrs['viewpointId'] = viewpoint_id
                outf[key].attrs['image_w'] = WIDTH
                outf[key].attrs['image_h'] = HEIGHT
                outf[key].attrs['vfov'] = VFOV

                num_finished_vps += 1
                progress_bar.update(num_finished_vps)

    progress_bar.finish()
    for process in processes:
        process.join()
            

class ImageFeaturesDB(object):
    def __init__(self, img_ft_file):
        self.image_feat_size = WIDTH*HEIGHT
        self.img_ft_file = img_ft_file
        self._feature_store = {}

    def get_image_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        if key in self._feature_store:
            ft = self._feature_store[key]
        else:
            with h5py.File(self.img_ft_file, 'r') as f:
                ft = f[key][...][:, :self.image_feat_size].astype(np.uint16)
                self._feature_store[key] = ft.reshape(VIEWPOINT_SIZE,WIDTH,HEIGHT,1)
        return ft

def read_features(args):
    
    imageDB = ImageFeaturesDB(args.output_file)
    scanvp_list = load_viewpoint_ids(args.connectivity_dir)

    for scan_id, viewpoint_id in scanvp_list:
        # Loop all discretized views from this location
        for ix in range(VIEWPOINT_SIZE):
            
            feature = imageDB.get_image_feature(scan_id,viewpoint_id)
            print(feature.shape)
            

            
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_file', default=None)
    parser.add_argument('--connectivity_dir', default='../datasets/R2R/connectivity')
    parser.add_argument('--scan_dir', default='../data/v1/scans')
    parser.add_argument('--output_file',)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    build_feature_file(args)
    #read_features(args)


