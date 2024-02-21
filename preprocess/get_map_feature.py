#!/usr/bin/env python3

''' Script to precompute map features using the CLIP-ViT-B/32 model, using 12 discretized horizontal views
    at each viewpoint in 30 degree increments, and the provided camera WIDTH, HEIGHT 
    and VFOV parameters. '''

import os
import sys

#import Matterport3DSimulator.MatterSim as MatterSim
import MatterSim
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
from tqdm import tqdm
from torch import optim

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip, RandomResizedCrop

from easydict import EasyDict as edict
from model_clip import CLIP

def BGR_to_RGB(cvimg):
    pilimg = cvimg.copy()
    pilimg[:, :, 0] = cvimg[:, :, 2]
    pilimg[:, :, 2] = cvimg[:, :, 0]
    return pilimg

clip_config = edict({
    'patches_grid': None,
    'patches_size': 16,
    'hidden_size': 768,
    'transformer_mlp_dim': 3072,
    'transformer_num_heads': 12,
    'transformer_num_layers': 12,
    'transformer_attention_dropout_rate': 0.,
    'transformer_dropout_rate': 0.
})

TSV_FIELDNAMES = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features']
VIEWPOINT_SIZE = 36 # Number of discretized views from one viewpoint
FEATURE_SIZE = 768

WIDTH = 224
HEIGHT = 224
VFOV = 60


def build_feature_extractor(checkpoint_file=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(False)
    model = CLIP(input_resolution=224, patch_size=clip_config.patches_size, width=clip_config.hidden_size, layers=clip_config.transformer_num_layers, heads=clip_config.transformer_num_heads).to(device)

    state_dict = torch.load(checkpoint_file, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)

   
    model.eval()

    img_transforms =  Compose([
            Resize((224,224), interpolation=Image.BICUBIC),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    return model, img_transforms, device

def build_simulator(connectivity_dir, scan_dir):
    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setDatasetPath(scan_dir)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setDepthEnabled(False)
    sim.setPreloadingEnabled(False)
    sim.setBatchSize(1)
    sim.initialize()
    return sim

def process_features(proc_id, out_queue, scanvp_list, args):
    print('start proc_id: %d' % proc_id)
    gpu_count = torch.cuda.device_count()
    local_rank = proc_id % gpu_count
    torch.cuda.set_device('cuda:{}'.format(local_rank))
    # Set up the simulator
    sim = build_simulator(args.connectivity_dir, args.scan_dir)

    # Set up PyTorch CNN model
    torch.set_grad_enabled(False)
    model, img_transforms, device = build_feature_extractor(args.checkpoint_file)

    for scan_id, viewpoint_id in scanvp_list:
        # Loop all discretized views from this location
        images = []
        for ix in range(VIEWPOINT_SIZE):
            if ix == 0:
                sim.newEpisode([scan_id], [viewpoint_id], [0], [math.radians(-30)])
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

            image = np.array(state.rgb, copy=True) # in BGR channel
            image = BGR_to_RGB(image)
            image = Image.fromarray(image) #cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)

        images = torch.stack([img_transforms(image).to(device) for image in images], 0)

        fts = []
        for k in range(0, len(images), args.batch_size):
            b_fts = model(images[k: k+args.batch_size])
            b_fts = b_fts.data.cpu().numpy().astype(np.float16)
            fts.append(b_fts)

        fts = np.concatenate(fts, 0)
        out_queue.put((scan_id, viewpoint_id, fts))

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
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_file', default=None)
    parser.add_argument('--connectivity_dir', default='../datasets/R2R/connectivity')
    parser.add_argument('--scan_dir', default='../data/v1/scans')
    parser.add_argument('--output_file')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    build_feature_file(args)


