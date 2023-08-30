'''
Instruction and trajectory (view and object features) dataset
'''
import os
import json
import jsonlines
import numpy as np
import h5py
import math
import MatterSim
from .common import load_nav_graphs
from .common import get_angle_fts, get_view_rel_angles
from .common import calculate_vp_rel_pos_fts
from .common import softmax
from PIL import Image
import random
MAX_DIST = 30   # normalize
MAX_STEP = 10   # normalize
TRAIN_MAX_STEP = 20

import json
from multiprocessing import Process
import multiprocessing
import imutils
import time
from collections import OrderedDict
import cv2
import h5py
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip, RandomResizedCrop



def load_viewpoint_ids(connectivity_dir):
    viewpoint_ids = []
    with open(os.path.join(connectivity_dir, 'scans.txt')) as f:
        scans = [x.strip() for x in f]
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json'%scan)) as f:
            data = json.load(f)
            viewpoint_ids.extend([(scan, x['image_id']) for x in data if x['included']])
    print('Loaded %d viewpoints' % len(viewpoint_ids))
    return viewpoint_ids

TSV_FIELDNAMES = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features']
VIEWPOINT_SIZE = 36 # Number of discretized views from one viewpoint

WIDTH = 128
HEIGHT = 128
VFOV = 60
GLOBAL_WIDTH = 14
GLOBAL_HEIGHT = 14

ERROR_MARGIN = 3.0



class DepthFeaturesDB(object):
    def __init__(self, img_ft_file):
        self.img_ft_file = img_ft_file
        self._feature_store = h5py.File(self.img_ft_file, 'r')

    def get_image_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)

        ft = self._feature_store[key][...][:].astype(np.uint16)

        return ft


class SemanticFeaturesDB(object):
    def __init__(self, img_ft_file):
        self.img_ft_file = img_ft_file
        self._feature_store = h5py.File(self.img_ft_file, 'r')

    def get_image_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        
        ft = self._feature_store[key][...][:].astype(np.float32)

        return ft

def get_rel_position(depth_map,angle):
    depth_y = depth_map.astype(np.float32) / 4000.
    depth_x = depth_y * (np.array([-6/7, -4/7, -2/7, 0., 2/7, 4/7, 6/7]*7,np.float32) * math.tan(math.pi/6))
    rel_x = depth_x * math.cos(angle) + depth_y * math.sin(angle)
    rel_y = depth_y * math.cos(angle) - depth_x * math.sin(angle)
    return rel_x, rel_y


class ReverieTextPathData(object):
    def __init__(
        self, anno_files, img_ft_file, obj_ft_file, scanvp_cands_file, connectivity_dir,
        image_feat_size=2048, image_prob_size=1000, angle_feat_size=4,
        obj_feat_size=None, obj_prob_size=None, max_objects=20,
        max_txt_len=100, in_memory=True, is_train=False, act_visited_node=False,semantic_map_dir="../datasets/R2R/features"
    ):
        self.is_train = is_train
        self.img_ft_file = img_ft_file
        self.obj_ft_file = obj_ft_file

        self.image_feat_size = image_feat_size
        self.image_prob_size = image_prob_size
        self.angle_feat_size = angle_feat_size
        self.obj_feat_size = obj_feat_size
        self.obj_prob_size = obj_prob_size

        self.obj_image_h = 480
        self.obj_image_w = 640
        self.obj_image_size = 480 * 640

        self.max_txt_len = max_txt_len
        self.max_objects = max_objects
        self.act_visited_node = act_visited_node

        self.in_memory = in_memory
        if self.in_memory:
            self._feature_store = {}

        # {scan_vp: {vp: [viewidx, rel_angle_dist, rel_heading, rel_elevation]}}
        self.scanvp_cands = json.load(open(scanvp_cands_file))

        self.graphs, self.shortest_distances, self.shortest_paths = load_nav_graphs(connectivity_dir)
        self.all_point_rel_angles = [get_view_rel_angles(baseViewId=i) for i in range(36)]
        self.all_point_angle_fts = [get_angle_fts(x[:, 0], x[:, 1], self.angle_feat_size) for x in self.all_point_rel_angles]

        self.data = []
        for anno_file in anno_files:
            with jsonlines.open(anno_file, 'r') as f:
                for item in f:
                    self.data.append(item)

        self.global_semantic = []
        self.global_position_x = []
        self.global_position_y = []
        self.global_mask = []
        self.max_x = -10000
        self.min_x = 10000
        self.max_y = -10000
        self.min_y = 10000
        self.heading = 0
        self.global_map = None

        self.DepthDB = DepthFeaturesDB(os.path.join(semantic_map_dir,"depth.hdf5"))
        self.SemanticDB = SemanticFeaturesDB(os.path.join(semantic_map_dir,"clip_p32.hdf5"))
        self.viewpoint_info = json.load(open(os.path.join(semantic_map_dir,"viewpoint_info.json")))
        self.cur_vp = None

        self.gt_path = None


    def __len__(self):
        return len(self.data)

    def get_scanvp_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        if self.is_train:
            ran = np.array([random.random() for i in range(36)])
            with h5py.File(self.img_ft_file+'/pth_vit_base_patch16_224_imagenet.hdf5', 'r') as f:
                    view_fts = f[key][...][:].astype(np.float32)

            with h5py.File(self.img_ft_file+'/aug_views.hdf5', 'r') as f:
                    aug_fts = f[key][...][:].astype(np.float32)
            view_fts[ran>0.5] = aug_fts[ran>0.5]
        else:
            with h5py.File(self.img_ft_file+'/pth_vit_base_patch16_224_imagenet.hdf5', 'r') as f:
                view_fts = f[key][...][:].astype(np.float32)

        obj_attrs = {}
        obj_fts = np.zeros((0, self.obj_feat_size+self.obj_prob_size), dtype=np.float32)
        if self.obj_ft_file is not None:
            with h5py.File(self.obj_ft_file, 'r') as f:
                if key in f:
                    obj_fts = f[key][...].astype(np.float32)
                    obj_fts = obj_fts[:self.max_objects]
                    for attr_key, attr_value in f[key].attrs.items():
                        if attr_key in ['directions', 'sizes', 'bboxes', 'obj_ids']:
                            obj_attrs[attr_key] = attr_value[:self.max_objects]
        if self.in_memory:
            self._feature_store[key] = (view_fts, obj_fts, obj_attrs)

        return view_fts, obj_fts, obj_attrs

    def get_obj_label(self, item, last_vp_objids):
        gt_obj_id = item['instr_id'].split('_')[1]
        for k, obj_id in enumerate(last_vp_objids):
            if obj_id == gt_obj_id:
                obj_label = k
                break
        else:
            # it occurs when the gt_objid is not in max_objects
            obj_label = -100 # ignore 
            # print('No groundtruth obj_id', item['instr_id'], len(obj_ids))
        return obj_label

    def get_act_labels(self, end_vp, item, gmap_vpids, gmap_visited_masks, traj_cand_vpids):
        scan = item['scan']
        pos_vps = item['pos_vps']
        if end_vp in pos_vps:
            global_act_label = local_act_label = 0
        else:
            global_act_label = local_act_label = -100
            # global: unvisited vp
            cand_min_dist = 100000000.
            for k, cand_vp in enumerate(gmap_vpids):
                if (k > 0) and (not gmap_visited_masks[k]):
                    min_dist = min([self.shortest_distances[scan][end_vp][cand_vp] \
                        + self.shortest_distances[scan][cand_vp][pos_vp] for pos_vp in pos_vps])
                    if min_dist < cand_min_dist:
                        cand_min_dist = min_dist
                        global_act_label = k # [stop] is 0
            # local: 
            cand_min_dist = 100000000.
            for k, cand_vp in enumerate(traj_cand_vpids[-1]):
                min_dist = min([self.shortest_distances[scan][end_vp][cand_vp] \
                    + self.shortest_distances[scan][cand_vp][pos_vp] for pos_vp in pos_vps])
                if min_dist < cand_min_dist:
                    cand_min_dist = min_dist
                    local_act_label = k + 1 # [stop] is 0
        return global_act_label, local_act_label

    def get_input(
        self, idx, end_vp_type, return_img_probs=False, return_act_label=False, 
        return_obj_label=False, end_vp=None
    ):
        item = self.data[idx]
        scan = item['scan']
        start_vp = item['path'][0]
        start_heading = item.get('heading', 0)
        self.cur_vp = start_vp
        self.heading = start_heading
        pos_vps = item['pos_vps']
        gt_path = item['path']
        
        if end_vp is None:
            if end_vp_type == 'pos':
                end_vp = pos_vps[np.random.randint(len(pos_vps))]
            elif end_vp_type == 'neg_in_gt_path':
                end_vps = [vp for vp in gt_path if vp not in pos_vps]
                if len(end_vps) == 0:
                    end_vps = gt_path
                end_vp = end_vps[np.random.randint(len(end_vps))]
            elif end_vp_type == 'neg_others':
                noneg_vp_set = set(pos_vps + gt_path)
                end_vps = [vp for vp in self.graphs[scan].nodes.keys() if vp not in noneg_vp_set]
                end_vp = end_vps[np.random.randint(len(end_vps))]

        gt_path = self.shortest_paths[scan][start_vp][end_vp]
        self.gt_path = gt_path
        cur_heading, cur_elevation = self.get_cur_angle(scan, gt_path, start_heading)

        if len(gt_path) > TRAIN_MAX_STEP:
            # truncate trajectory
            gt_path = gt_path[:TRAIN_MAX_STEP] + [end_vp]
            

        traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, \
            last_vp_angles, last_vp_objids, grid_fts,grid_map,gridmap_pos_fts, target_patch_id = self.get_traj_pano_fts(scan, gt_path)



        # global: the first token is [stop]
        gmap_vpids, gmap_step_ids, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists = \
            self.get_gmap_inputs(scan, gt_path, cur_heading, cur_elevation)

        # local: the first token is [stop]
        vp_pos_fts = self.get_vp_pos_fts(scan, start_vp, end_vp,
            traj_cand_vpids[-1], cur_heading, cur_elevation, len(traj_nav_types[-1]))

        outs = {
            'instr_id': item['instr_id'],
            'instr_encoding': item['instr_encoding'][:self.max_txt_len],
            
            'traj_view_img_fts': [x[:, :self.image_feat_size] for x in traj_view_img_fts],
            'traj_obj_img_fts': [x[:, :self.obj_feat_size] for x in traj_obj_img_fts],
            'traj_loc_fts': traj_loc_fts,
            'traj_nav_types': traj_nav_types,
            'traj_cand_vpids': traj_cand_vpids,
            'traj_vpids': gt_path,

            'gmap_vpids': gmap_vpids,
            'gmap_step_ids': gmap_step_ids,
            'gmap_visited_masks': gmap_visited_masks,
            'gmap_pos_fts': gmap_pos_fts,
            'gmap_pair_dists': gmap_pair_dists,

            'vp_pos_fts': vp_pos_fts,
            # 'vp_objids': last_vp_objids,
            'vp_angles': last_vp_angles,
            'grid_fts':grid_fts,
            'grid_map':grid_map,
            'gridmap_pos_fts': gridmap_pos_fts,
            'target_patch_id':target_patch_id
        }

        if return_obj_label:
            outs['obj_labels'] = self.get_obj_label(item, last_vp_objids)

        if return_act_label:
            global_act_label, local_act_label = self.get_act_labels(
                end_vp, item, gmap_vpids, gmap_visited_masks, traj_cand_vpids
            )
            outs['global_act_labels'] = global_act_label
            outs['local_act_labels'] = local_act_label

        if return_img_probs:
            # TODO: whether adding gmap img probs
            outs['vp_view_probs'] = softmax(traj_view_img_fts[-1][:, self.image_feat_size:], dim=1)
            outs['vp_obj_probs'] = softmax(traj_obj_img_fts[-1][:, self.obj_feat_size:], dim=1)

        return outs

    def get_cur_angle(self, scan, path, start_heading):
        if len(path) < 2:
            heading = start_heading
            elevation = 0
        else:
            prev_vp = path[-2]
            cur_vp = path[-1]
            viewidx = self.scanvp_cands['%s_%s'%(scan, prev_vp)][cur_vp][0]
            heading = (viewidx % 12) * math.radians(30)
            elevation = (viewidx // 12 - 1) * math.radians(30)
        return heading, elevation


    def get_gridmap_pos_fts(self, half_len):
        rel_angles, rel_dists = [], []
        center_position = [0.,0.,0.]

        cell_len = half_len*2 / GLOBAL_WIDTH
        for i in range(GLOBAL_WIDTH):
            for j in range(GLOBAL_HEIGHT):
                position = [0.,0.,0.]
                position[0] = i*cell_len - half_len + cell_len/2.
                position[1] = j*cell_len - half_len + cell_len/2.
                position[2] = 0.
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(center_position, position, base_heading=0., base_elevation=0.)
                rel_angles.append([rel_heading, rel_elevation])
                rel_dists.append(
                    [rel_dist / MAX_DIST]
                )


        rel_angles = np.array(rel_angles).astype(np.float32)
        rel_dists = np.array(rel_dists).astype(np.float32)
        rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1])
        gridmap_pos_fts = np.concatenate([rel_ang_fts, rel_dists], 1)

        return gridmap_pos_fts


    def getGlobalMap(self,scan_id,viewpoint_id):


        viewpoint_x_list = []
        viewpoint_y_list = []
        depth = self.DepthDB.get_image_feature(scan_id,viewpoint_id)
        patch_center_index = np.array([9+i*18 for i in range(7)])

        depth = depth[:,patch_center_index][:,:,patch_center_index].reshape(36,-1)

        depth_mask = np.ones(depth.shape)
        depth_mask[depth==0] = 0
        self.global_mask.append(depth_mask[12:24].reshape(12,-1))
        position = self.viewpoint_info['%s_%s' % (scan_id, viewpoint_id)]

        cur_step_id = len(self.global_mask) - 1
        next_id = cur_step_id
        if cur_step_id < len(self.gt_path)-1:
            next_id = cur_step_id + 1
        
        target_position_x = self.viewpoint_info['%s_%s' % (scan_id, self.gt_path[next_id])]["x"] - position["x"]
        target_position_y = self.viewpoint_info['%s_%s' % (scan_id, self.gt_path[next_id])]["y"] - position["y"]


        
        for ix in range(12,24):
            rel_x, rel_y = get_rel_position(depth[ix:ix+1],(ix-12)*math.pi/6)  
            global_x = rel_x + position["x"]
            global_y = rel_y + position["y"]
            viewpoint_x_list.append(global_x)
            viewpoint_y_list.append(global_y)



        semantic = self.SemanticDB.get_image_feature(scan_id,viewpoint_id)

        
        if self.global_semantic == []:
            self.global_semantic = semantic[:,1:]
            self.global_map = np.zeros((12*49,))
            
        else:
            self.global_semantic = semantic[:,1:]
            self.global_map =  np.concatenate([self.global_map,np.zeros((12*49,))],0)

        self.global_map.fill(-1)
        position_x = np.concatenate(viewpoint_x_list,0)
        position_y = np.concatenate(viewpoint_y_list,0)
        self.global_position_x.append(position_x)
        self.global_position_y.append(position_y)

        tmp_max_x = position_x.max()
        if tmp_max_x > self.max_x: self.max_x = tmp_max_x
        tmp_min_x = position_x.min()
        if tmp_min_x < self.min_x: self.min_x = tmp_min_x
        tmp_max_y = position_y.max()
        if tmp_max_y > self.max_y: self.max_y = tmp_max_y
        tmp_min_y = position_y.min()
        if tmp_min_y < self.min_y: self.min_y = tmp_min_y


        if position["x"]-self.min_x > self.max_x-position["x"] : x_half_len = position["x"]-self.min_x
        else: x_half_len = self.max_x-position["x"]

        if position["y"]-self.min_y > self.max_y-position["y"] : y_half_len = position["y"]-self.min_y
        else: y_half_len = self.max_y-position["y"]

        if x_half_len > y_half_len : half_len = x_half_len
        else: half_len = y_half_len

        half_len = half_len * 2/3
        min_x = position["x"] - half_len
        max_x = position["x"] + half_len
        min_y = position["y"] - half_len
        max_y = position["y"] + half_len

        angle = -self.heading
        sRotatex = target_position_x * math.cos(angle) + target_position_y * math.sin(angle)
        sRotatey = target_position_y * math.cos(angle) - target_position_x * math.sin(angle)

        target_patch_x = int((sRotatex + half_len)*14 // (2*half_len))
        target_patch_y = int((sRotatey + half_len)*14 // (2*half_len))
        target_patch_x = min(max(target_patch_x,0),13)
        target_patch_y = min(max(target_patch_y,0),13)
        target_patch_id = 1 + target_patch_x*14 + target_patch_y

        if next_id == cur_step_id:
            target_patch_id = 0
        
        global_position_x = np.concatenate(self.global_position_x,0)
        global_position_y = np.concatenate(self.global_position_y,0)
        local_map = self.global_semantic
        global_mask = np.concatenate(self.global_mask,0)

        tmp_x = global_position_x - position["x"]
        tmp_y = global_position_y - position["y"]

        map_x = tmp_x * math.cos(angle) + tmp_y * math.sin(angle)
        map_y = tmp_y * math.cos(angle) - tmp_x * math.sin(angle)
        map_x = ((map_x + half_len) / (2*half_len) * (GLOBAL_WIDTH-1)).astype(np.int32)

        map_y = ((map_y + half_len) / (2*half_len) * (GLOBAL_HEIGHT-1)).astype(np.int32)

        map_x[map_x<0] = 0
        map_x[map_x>=GLOBAL_WIDTH] = GLOBAL_WIDTH-1

        map_y[map_y<0] = 0
        map_y[map_y>=GLOBAL_HEIGHT] = GLOBAL_HEIGHT-1

        label_index = (global_mask==1)

        map_index = map_x*14 + map_y
        map_index = map_index.reshape(-1)
        label_index = label_index.reshape(-1)
        
        for patch_id in range(GLOBAL_WIDTH*GLOBAL_HEIGHT):

            filter_index = (map_index==patch_id)&label_index

            self.global_map[filter_index] = patch_id

        gridmap_pos_fts = self.get_gridmap_pos_fts(half_len)
        return self.global_semantic,self.global_position_x,self.global_position_y,self.global_mask,self.global_map,self.max_x,self.min_x,self.max_y,self.min_y, gridmap_pos_fts, target_patch_id



    def get_traj_pano_fts(self, scan, path):
        '''
        Tokens in each pano: [cand_views, noncand_views, objs]
        Each token consists of (img_fts, loc_fts (ang_fts, box_fts), nav_types)
        '''

        self.global_semantic = []
        self.global_position_x = []
        self.global_position_y = []
        self.global_mask = []
        self.max_x = -10000
        self.min_x = 10000
        self.max_y = -10000
        self.min_y = 10000
        self.global_map = None

        traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids = [], [], [], [], []
        grid_map = None
        grid_fts = np.array([])
        for vp in path:
            if self.cur_vp != vp and vp in self.scanvp_cands['%s_%s'%(scan, self.cur_vp)]:
                viewidx = self.scanvp_cands['%s_%s'%(scan, self.cur_vp)][vp][0]          
                self.heading = (viewidx % 12) * math.radians(30)
            self.cur_vp = vp
            self.global_semantic,self.global_position_x,self.global_position_y,self.global_mask,self.global_map,self.max_x,self.min_x,self.max_y,self.min_y,gridmap_pos_fts, target_patch_id = self.getGlobalMap(scan, vp)
            
            if grid_fts.shape == (0,):
                grid_fts = self.global_semantic
            else:
                grid_fts = np.concatenate((grid_fts,self.global_semantic),axis=0)
            grid_map = self.global_map

            view_fts, obj_img_fts, obj_attrs = self.get_scanvp_feature(scan, vp)
            
            view_img_fts, view_angles, cand_vpids = [], [], []
            # cand views
            nav_cands = self.scanvp_cands['%s_%s'%(scan, vp)]
            used_viewidxs = set()
            for k, v in nav_cands.items():
                used_viewidxs.add(v[0])
                view_img_fts.append(view_fts[v[0]])
                # TODO: whether using correct heading at each step
                view_angle = self.all_point_rel_angles[12][v[0]]
                view_angles.append([view_angle[0] + v[2], view_angle[1] + v[3]])
                cand_vpids.append(k)
            # non cand views
            view_img_fts.extend([view_fts[idx] for idx in range(36) if idx not in used_viewidxs])
            view_angles.extend([self.all_point_rel_angles[12][idx] for idx in range(36) if idx not in used_viewidxs])
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)    # (n_views, dim_ft)
            view_angles = np.stack(view_angles, 0)
            view_ang_fts = get_angle_fts(view_angles[:, 0], view_angles[:, 1], self.angle_feat_size)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            
            # object features
            num_objs = obj_img_fts.shape[0]
            obj_angles = np.zeros((num_objs, 2), dtype=np.float32)
            obj_ang_fts = np.zeros((num_objs, self.angle_feat_size), dtype=np.float32)
            obj_box_fts = np.zeros((num_objs, 3), dtype=np.float32)
            if num_objs > 0:
                for k, (w, h) in enumerate(obj_attrs['sizes']):
                    obj_angles[k] = obj_attrs['directions'][k]
                    obj_box_fts[k] = [h/self.obj_image_h, w/self.obj_image_w, (h*w)/self.obj_image_size]           
                obj_ang_fts = get_angle_fts(obj_angles[:, 0], obj_angles[:, 1], self.angle_feat_size)

            # combine pano features
            traj_view_img_fts.append(view_img_fts)
            traj_obj_img_fts.append(obj_img_fts)
            traj_loc_fts.append(
                np.concatenate(
                    [np.concatenate([view_ang_fts, view_box_fts], 1),
                     np.concatenate([obj_ang_fts, obj_box_fts], 1)], axis=0
                )
            )
            traj_nav_types.append(
                [1] * len(cand_vpids) + [0] * (36 - len(used_viewidxs)) + [2] * len(obj_img_fts)
            )
            traj_cand_vpids.append(cand_vpids)

            last_vp_objids = obj_attrs.get('obj_ids', [])
            last_vp_angles = np.concatenate([view_angles, obj_angles], 0)

        return traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, \
               last_vp_angles, last_vp_objids, grid_fts.reshape((-1,768)),grid_map, gridmap_pos_fts,target_patch_id
        
    def get_gmap_inputs(self, scan, path, cur_heading, cur_elevation):
        scan_graph = self.graphs[scan]
        cur_vp = path[-1]

        visited_vpids, unvisited_vpids = {}, {}
        for t, vp in enumerate(path):
            visited_vpids[vp] = t + 1
            if vp in unvisited_vpids:
                del unvisited_vpids[vp]
            for next_vp in self.scanvp_cands['%s_%s'%(scan, vp)].keys():
                if next_vp not in visited_vpids:
                    unvisited_vpids[next_vp] = 0
        # add [stop] token
        gmap_vpids = [None] + list(visited_vpids.keys()) + list(unvisited_vpids.keys())
        gmap_step_ids = [0] + list(visited_vpids.values()) + list(unvisited_vpids.values())
        if self.act_visited_node:
            gmap_visited_masks = [0]
            for vp in gmap_vpids[1:]:
                if vp == path[-1]:
                    gmap_visited_masks.append(1)
                else:
                    gmap_visited_masks.append(0)
        else:
            gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)

        # shape=(num_gmap_vpids, 7)
        gmap_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, gmap_vpids, cur_heading, cur_elevation)
        
        gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
        for i in range(1, len(gmap_vpids)):
            for j in range(i+1, len(gmap_vpids)):
                gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                    self.shortest_distances[scan][gmap_vpids[i]][gmap_vpids[j]]

        return gmap_vpids, gmap_step_ids, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists
    
    def get_gmap_pos_fts(self, scan, cur_vp, gmap_vpids, cur_heading, cur_elevation):
        # dim=7 (sin(heading), cos(heading), sin(elevation), cos(elevation),
        #  line_dist, shortest_dist, shortest_step)
        rel_angles, rel_dists = [], []
        for vp in gmap_vpids:
            if vp is None:
                rel_angles.append([0, 0])
                rel_dists.append([0, 0, 0])
            else:
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                    self.graphs[scan].nodes[cur_vp]['position'], 
                    self.graphs[scan].nodes[vp]['position'],
                    base_heading=cur_heading, base_elevation=cur_elevation,
                )
                rel_angles.append([rel_heading, rel_elevation])
                rel_dists.append(
                    [rel_dist / MAX_DIST, self.shortest_distances[scan][cur_vp][vp] / MAX_DIST, \
                    (len(self.shortest_paths[scan][cur_vp][vp]) - 1) / MAX_STEP]
                )
        rel_angles = np.array(rel_angles).astype(np.float32)
        rel_dists = np.array(rel_dists).astype(np.float32)
        rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1], self.angle_feat_size)
        return np.concatenate([rel_ang_fts, rel_dists], 1)
        
    def get_vp_pos_fts(self, scan, start_vp, cur_vp, cand_vpids, cur_heading, cur_elevation, vp_ft_len):
        cur_cand_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, cand_vpids, cur_heading, cur_elevation)
        cur_start_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, [start_vp], cur_heading, cur_elevation)
                
        # add [stop] token at beginning
        vp_pos_fts = np.zeros((vp_ft_len+1, 14), dtype=np.float32)
        vp_pos_fts[:, :7] = cur_start_pos_fts
        vp_pos_fts[1:len(cur_cand_pos_fts)+1, 7:] = cur_cand_pos_fts

        return vp_pos_fts
       

class R2RTextPathData(ReverieTextPathData):
    def __init__(
        self, anno_files, img_ft_file, scanvp_cands_file, connectivity_dir,
        image_feat_size=2048, image_prob_size=1000, angle_feat_size=4,
        max_txt_len=100, in_memory=True, is_train=False, act_visited_node=False
    ):
        super().__init__(
            anno_files, img_ft_file, None, scanvp_cands_file, connectivity_dir,
            image_feat_size=image_feat_size, image_prob_size=image_prob_size,
            angle_feat_size=angle_feat_size, obj_feat_size=0, obj_prob_size=0, 
            max_objects=0, max_txt_len=max_txt_len, in_memory=in_memory, is_train=is_train,
            act_visited_node=act_visited_node
        )

    def get_scanvp_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        if self.is_train:
            ran = np.array([random.random() for i in range(36)])
            with h5py.File(self.img_ft_file+'/pth_vit_base_patch16_224_imagenet.hdf5', 'r') as f:
                    view_fts = f[key][...][:].astype(np.float32)

            with h5py.File(self.img_ft_file+'/aug_views.hdf5', 'r') as f:
                    aug_fts = f[key][...][:].astype(np.float32)
            view_fts[ran>0.5] = aug_fts[ran>0.5]
        else:
            with h5py.File(self.img_ft_file+'/pth_vit_base_patch16_224_imagenet.hdf5', 'r') as f:
                view_fts = f[key][...][:].astype(np.float32)
        
        return view_fts

    def get_act_labels(self, end_vp, end_idx, item, gmap_vpids, traj_cand_vpids):
        if end_vp == item['path'][-1]:  # stop
            global_act_label = local_act_label = 0
        else:
            global_act_label = local_act_label = -100
            # global: unvisited vp
            gt_next_vp = item['path'][end_idx + 1]
            for k, cand_vp in enumerate(gmap_vpids):
                if cand_vp == gt_next_vp:
                    global_act_label = k
                    break
            # local: 
            for k, cand_vp in enumerate(traj_cand_vpids[-1]):
                if cand_vp == gt_next_vp:
                    local_act_label = k + 1 # [stop] is 0
                    break
        return global_act_label, local_act_label

    def get_input(
        self, idx, end_vp_type, return_img_probs=False, return_act_label=False, end_vp=None
    ):
        item = self.data[idx]
        scan = item['scan']
        start_vp = item['path'][0]
        self.cur_vp = start_vp
        start_heading = item['heading']
        self.heading = start_heading
        gt_path = item['path']
        self.gt_path = gt_path
        if end_vp is None:
            if end_vp_type == 'pos': 
                # name convention with REVERIE (last vp)
                end_idx = len(gt_path) - 1
                end_vp = gt_path[-1]
            elif end_vp_type in ['neg_in_gt_path', 'neg_others']:
                # name convention with REVERIE (mid vps in the path)
                end_vps = gt_path[:-1]
                end_idx = np.random.randint(len(end_vps))
                end_vp = end_vps[end_idx]
        else:
            assert end_vp in gt_path
            end_idx = gt_path.index(end_vp)
            
        gt_path = gt_path[:end_idx+1]
        cur_heading, cur_elevation = self.get_cur_angle(scan, gt_path, start_heading)

        if len(gt_path) > TRAIN_MAX_STEP:
            # truncate trajectory
            gt_path = gt_path[:TRAIN_MAX_STEP] + [end_vp]
            

        traj_view_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, \
            last_vp_angles,grid_fts,grid_map,gridmap_pos_fts, target_patch_id = self.get_traj_pano_fts(scan, gt_path)

        # global: the first token is [stop]
        gmap_vpids, gmap_step_ids, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists = \
            self.get_gmap_inputs(scan, gt_path, cur_heading, cur_elevation)

        # local: the first token is [stop]
        vp_pos_fts = self.get_vp_pos_fts(scan, start_vp, end_vp,
            traj_cand_vpids[-1], cur_heading, cur_elevation, len(traj_nav_types[-1]))

        outs = {
            'instr_id': item['instr_id'],
            'instr_encoding': item['instr_encoding'][:self.max_txt_len],
            
            'traj_view_img_fts': [x[:, :self.image_feat_size] for x in traj_view_img_fts],
            'traj_loc_fts': traj_loc_fts,
            'traj_nav_types': traj_nav_types,
            'traj_cand_vpids': traj_cand_vpids,
            'traj_vpids': gt_path,

            'gmap_vpids': gmap_vpids,
            'gmap_step_ids': gmap_step_ids,
            'gmap_visited_masks': gmap_visited_masks,
            'gmap_pos_fts': gmap_pos_fts,
            'gmap_pair_dists': gmap_pair_dists,

            'vp_pos_fts': vp_pos_fts,
            'vp_angles': last_vp_angles,
            'grid_fts':grid_fts,
            'grid_map':grid_map,
            'gridmap_pos_fts': gridmap_pos_fts,
            'target_patch_id':target_patch_id
        }

        if return_act_label:
            global_act_label, local_act_label = self.get_act_labels(
                end_vp, end_idx, item, gmap_vpids, traj_cand_vpids
            )
            outs['global_act_labels'] = global_act_label
            outs['local_act_labels'] = local_act_label

        if return_img_probs:
            # TODO: whether adding gmap img probs
            outs['vp_view_probs'] = softmax(traj_view_img_fts[-1][:, self.image_feat_size:], dim=1)

        return outs


        

    def get_traj_pano_fts(self, scan, path):
        '''
        Tokens in each pano: [cand_views, noncand_views, objs]
        Each token consists of (img_fts, loc_fts (ang_fts, box_fts), nav_types)
        '''
        
        self.global_semantic = []
        self.global_position_x = []
        self.global_position_y = []
        self.global_mask = []
        self.max_x = -10000
        self.min_x = 10000
        self.max_y = -10000
        self.min_y = 10000


        traj_view_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids = [], [], [], []
        grid_map = None
        grid_fts = np.array([])

        for vp in path:

            if self.cur_vp != vp and vp in self.scanvp_cands['%s_%s'%(scan, self.cur_vp)]:
                viewidx = self.scanvp_cands['%s_%s'%(scan, self.cur_vp)][vp][0]          
                self.heading = (viewidx % 12) * math.radians(30)
            self.cur_vp = vp
            self.global_semantic,self.global_position_x,self.global_position_y,self.global_mask,self.global_map,self.max_x,self.min_x,self.max_y,self.min_y,gridmap_pos_fts, target_patch_id = self.getGlobalMap(scan, vp)
            
            if grid_fts.shape == (0,):
                grid_fts = self.global_semantic
            else:
                grid_fts = np.concatenate((grid_fts,self.global_semantic),axis=0)
            grid_map = self.global_map

            view_fts = self.get_scanvp_feature(scan, vp)

            view_img_fts, view_angles, cand_vpids = [], [], []
            # cand views
            nav_cands = self.scanvp_cands['%s_%s'%(scan, vp)]
            used_viewidxs = set()
            for k, v in nav_cands.items():
                used_viewidxs.add(v[0])
                view_img_fts.append(view_fts[v[0]])
                # TODO: whether using correct heading at each step
                view_angle = self.all_point_rel_angles[12][v[0]]
                view_angles.append([view_angle[0] + v[2], view_angle[1] + v[3]])
                cand_vpids.append(k)
            # non cand views
            view_img_fts.extend([view_fts[idx] for idx in range(36) if idx not in used_viewidxs])
            view_angles.extend([self.all_point_rel_angles[12][idx] for idx in range(36) if idx not in used_viewidxs])
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)    # (n_views, dim_ft)
            view_angles = np.stack(view_angles, 0)
            view_ang_fts = get_angle_fts(view_angles[:, 0], view_angles[:, 1], self.angle_feat_size)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            
            # combine pano features
            traj_view_img_fts.append(view_img_fts)
            traj_loc_fts.append(np.concatenate([view_ang_fts, view_box_fts], 1))
            traj_nav_types.append([1] * len(cand_vpids) + [0] * (36 - len(used_viewidxs)))
            traj_cand_vpids.append(cand_vpids)
            
            last_vp_angles = view_angles

        return traj_view_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, last_vp_angles, grid_fts.reshape((-1,768)),grid_map, gridmap_pos_fts, target_patch_id
     

class SoonTextPathData(ReverieTextPathData):
    def __init__(
        self, anno_files, img_ft_file, obj_ft_file, scanvp_cands_file, connectivity_dir,
        image_feat_size=2048, image_prob_size=1000, angle_feat_size=4,
        obj_feat_size=None, obj_prob_size=None, max_objects=20,
        max_txt_len=100, in_memory=True, is_train=False, act_visited_node=False
    ):
        super().__init__(
            anno_files, img_ft_file, obj_ft_file, scanvp_cands_file, connectivity_dir,
            image_feat_size=image_feat_size, image_prob_size=image_prob_size,
            angle_feat_size=angle_feat_size, obj_feat_size=obj_feat_size, 
            obj_prob_size=obj_prob_size, max_objects=max_objects, 
            max_txt_len=max_txt_len, in_memory=in_memory,is_train=is_train,
            act_visited_node=act_visited_node
        )
        self.obj_image_h = self.obj_image_w = 600
        self.obj_image_size = 600 * 600

    def get_scanvp_feature(self, scan, viewpoint):

        key = '%s_%s' % (scan, viewpoint)
        
        if self.is_train:
            ran = np.array([random.random() for i in range(36)])
            with h5py.File(self.img_ft_file+'/pth_vit_base_patch16_224_imagenet.hdf5', 'r') as f:
                    view_fts = f[key][...][:].astype(np.float32)

            with h5py.File(self.img_ft_file+'/aug_views.hdf5', 'r') as f:
                    aug_fts = f[key][...][:].astype(np.float32)
            view_fts[ran>0.5] = aug_fts[ran>0.5]
        else:
            with h5py.File(self.img_ft_file+'/pth_vit_base_patch16_224_imagenet.hdf5', 'r') as f:
                view_fts = f[key][...][:].astype(np.float32)
        
        obj_attrs = {}
        obj_fts = np.zeros((0, self.obj_feat_size+self.obj_prob_size), dtype=np.float32)
        if self.obj_ft_file is not None:
            with h5py.File(self.obj_ft_file, 'r') as f:
                if key in f:
                    obj_fts = f[key][...].astype(np.float32)
                    obj_fts = obj_fts[:self.max_objects]
                    for attr_key, attr_value in f[key].attrs.items():
                        if attr_key in ['directions', 'bboxes', 'obj_ids']:
                            obj_attrs[attr_key] = attr_value[:self.max_objects]
                    obj_attrs['bboxes'] = np.array(obj_attrs['bboxes']).astype(np.float32)
                    obj_attrs['sizes'] = np.zeros((len(obj_attrs['bboxes']), 2), dtype=np.float32)
                    obj_attrs['sizes'][:, 0] = obj_attrs['bboxes'][:, 2] - obj_attrs['bboxes'][:, 0]
                    obj_attrs['sizes'][:, 1] = obj_attrs['bboxes'][:, 3] - obj_attrs['bboxes'][:, 1]
        if self.in_memory:
            self._feature_store[key] = (view_fts, obj_fts, obj_attrs)

        return view_fts, obj_fts, obj_attrs

    def get_obj_label(self, item, last_vp_objids):
        obj_label = item['obj_pseudo_label']['idx']
        if obj_label >= self.max_objects:
            obj_label = -100
        return obj_label

    def get_input(
        self, idx, end_vp_type, return_img_probs=False, return_act_label=False, 
        return_obj_label=False, end_vp=None
    ):
        if end_vp_type == 'pos':
            end_vp = self.data[idx]['path'][-1]
        return super().get_input(
            idx, end_vp_type, 
            return_img_probs=return_img_probs, 
            return_act_label=return_act_label, 
            return_obj_label=return_obj_label, 
            end_vp=end_vp
        )
