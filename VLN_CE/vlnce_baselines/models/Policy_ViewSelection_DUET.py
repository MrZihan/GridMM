import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from gym import Space
from habitat import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo.policy import Net

from vlnce_baselines.models.duet.vlnbert_init import get_vlnbert_models
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.models.encoders.instruction_encoder import (
    InstructionEncoder,
)
from vlnce_baselines.models.encoders.resnet_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder,
)
from vlnce_baselines.models.policy import ILPolicy

from waypoint_prediction.utils import nms
from vlnce_baselines.models.utils import (
    angle_feature_with_ele, dir_angle_feature_with_ele, length2mask, pad_tensors, gen_seq_masks, get_angle_fts, get_angle_feature, get_point_angle_feature, calculate_vp_rel_pos_fts, calc_position_distance , pad_tensors_wgrad)
import math
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip, RandomResizedCrop
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.nn.utils.rnn import pad_sequence



@baseline_registry.register_policy
class PolicyViewSelectionDUET(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        model_config: Config,
    ):
        super().__init__(
            DUET(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
            ),
            action_space.n,
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: Space, action_space: Space
    ):
        config.defrost()
        config.MODEL.TORCH_GPU_ID = config.TORCH_GPU_IDS[config.local_rank]
        config.freeze()

        return cls(
            observation_space=observation_space,
            action_space=action_space,
            model_config=config.MODEL,
        )


class DUET(Net):
    def __init__(
        self, observation_space: Space, model_config: Config, num_actions,
    ):
        super().__init__()

        device = (
            torch.device("cuda", model_config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.device = device

        print('\nInitalizing the DUET model ...')
        self.vln_bert = get_vlnbert_models(config=None)
        self.vln_bert.config.directions = 1  # a trivial number, change during nav
        layer_norm_eps = self.vln_bert.config.layer_norm_eps

        # Init the depth encoder
        assert model_config.DEPTH_ENCODER.cnn_type in [
            "VlnResnetDepthEncoder"
        ], "DEPTH_ENCODER.cnn_type must be VlnResnetDepthEncoder"
        self.depth_encoder = VlnResnetDepthEncoder(
            observation_space,
            output_size=model_config.DEPTH_ENCODER.output_size,
            checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
            backbone=model_config.DEPTH_ENCODER.backbone,
            spatial_output=model_config.spatial_output,
        )

        # Init the RGB encoder
        assert model_config.RGB_ENCODER.cnn_type in [
            "TorchVisionResNet152", "TorchVisionResNet50"
        ], "RGB_ENCODER.cnn_type must be TorchVisionResNet152 or TorchVisionResNet50"
        if model_config.RGB_ENCODER.cnn_type == "TorchVisionResNet50":
            self.rgb_encoder = TorchVisionResNet50(
                observation_space,
                model_config.RGB_ENCODER.output_size,
                device,
                spatial_output=model_config.spatial_output,
            )

        # merging visual inputs
        self.space_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(start_dim=2),)
        self.rgb_linear = nn.Sequential(
            nn.Linear(
                model_config.RGB_ENCODER.encode_size,
                model_config.RGB_ENCODER.output_size,
            ),
            nn.ReLU(True),
        )
        self.depth_linear = nn.Sequential(
            nn.Linear(
                model_config.DEPTH_ENCODER.encode_size,
                model_config.DEPTH_ENCODER.output_size,
            ),
            nn.ReLU(True),
        )
        self.vismerge_linear = nn.Sequential(
            nn.Linear(
                model_config.DEPTH_ENCODER.output_size + model_config.RGB_ENCODER.output_size + model_config.VISUAL_DIM.directional,
                model_config.VISUAL_DIM.vis_hidden,
            ),
            nn.ReLU(True),
        )

        self.action_state_project = nn.Sequential(
            nn.Linear(model_config.VISUAL_DIM.vis_hidden+model_config.VISUAL_DIM.directional,
            model_config.VISUAL_DIM.vis_hidden),
            nn.Tanh())
        self.action_LayerNorm = BertLayerNorm(
            model_config.VISUAL_DIM.vis_hidden, eps=layer_norm_eps)

        self.drop_env = nn.Dropout(p=0.4)

        self.view_transforms =  Compose([
            Resize((224,224), interpolation=Image.BICUBIC),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        batch_size = model_config.batch_size

        self.headings = [0 for i in range(batch_size)]
        self.positions = None

        self.start_positions = None
        self.start_headings = None
        self.action_step = 0
        self.traj_embeds = [[] for i in range(batch_size)]
        self.traj_map = [[] for i in range(batch_size)]

        self.train()

    @property  # trivial argument, just for init with habitat
    def output_size(self):
        return 1

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return 1


    def forward(self, mode=None, 
            waypoint_predictor=None,
            observations=None,
            lang_idx_tokens=None, lang_masks=None,
            lang_feats=None, lang_token_type_ids=None,
            headings=None,  positions=None,
            cand_rgb=None, cand_depth=None,
            cand_direction=None, cand_mask=None, candidate_lengths=None, batch_angles=None, batch_distances=None,
            masks=None, batch_view_img_fts=None, batch_loc_fts=None, batch_nav_types=None, batch_view_lens=None, 
            post_states=None, in_train=True):


        if mode == 'language':
            language_features = self.vln_bert(
                'language', (lang_idx_tokens,lang_masks))

            return language_features

        elif mode == 'waypoint':

            batch_size = observations['instruction'].size(0)
            ''' encoding rgb/depth at all directions ----------------------------- '''
            NUM_ANGLES = 120    # 120 angles 3 degrees each
            NUM_IMGS = 12
            NUM_CLASSES = 12    # 12 distances at each sector
            depth_batch = torch.zeros_like(observations['depth']).repeat(NUM_IMGS, 1, 1, 1)
            rgb_batch = torch.zeros_like(observations['rgb']).repeat(NUM_IMGS, 1, 1, 1)

            # reverse the order of input images to clockwise
            # single view images in clockwise agrees with the panoramic image
            a_count = 0
            for i, (k, v) in enumerate(observations.items()):
                if 'depth' in k:
                    for bi in range(v.size(0)):
                        ra_count = (NUM_IMGS - a_count)%NUM_IMGS
                        depth_batch[ra_count+bi*NUM_IMGS] = v[bi]
                        rgb_batch[ra_count+bi*NUM_IMGS] = observations[k.replace('depth','rgb')][bi]
                    a_count += 1
            obs_view12 = {}
            obs_view12['depth'] = depth_batch
            obs_view12['rgb'] = rgb_batch

            depth_embedding = self.depth_encoder(obs_view12)
            rgb_embedding = self.rgb_encoder(obs_view12)

            image_list = [Image.fromarray(rgb_batch[img_id].cpu().numpy()) for img_id in range(rgb_batch.shape[0])]
            
            batch_view_fts = np.concatenate([self.view_transforms(image).unsqueeze(0) for image in image_list], 0)
            batch_view_fts = torch.tensor(batch_view_fts).to(self.device)

            with torch.no_grad():
                batch_view_fts = self.vln_bert.visual_encoder.forward_features(batch_view_fts)[:,0,:]

            batch_view_fts = batch_view_fts.view(batch_size,12,768).cpu().numpy()
  


            ''' waypoint prediction ----------------------------- '''
            waypoint_heatmap_logits = waypoint_predictor(
                rgb_embedding, depth_embedding)


            # from heatmap to points
            batch_x_norm = torch.softmax(
                waypoint_heatmap_logits.reshape(
                    batch_size, NUM_ANGLES*NUM_CLASSES,
                ), dim=1
            )
            batch_x_norm = batch_x_norm.reshape(
                batch_size, NUM_ANGLES, NUM_CLASSES,
            )
            batch_x_norm_wrap = torch.cat((
                batch_x_norm[:,-1:,:], 
                batch_x_norm, 
                batch_x_norm[:,:1,:]), 
                dim=1)
            batch_output_map = nms(
                batch_x_norm_wrap.unsqueeze(1), 
                max_predictions=5,
                sigma=(7.0,5.0))

            # predicted waypoints before sampling
            batch_output_map = batch_output_map.squeeze(1)[:,1:-1,:]

            candidate_lengths = ((batch_output_map!=0).sum(-1).sum(-1) + 1).tolist()
            if isinstance(candidate_lengths, int):
                candidate_lengths = [candidate_lengths]
            max_candidate = max(candidate_lengths)  # including stop
            cand_mask = length2mask(candidate_lengths, device=self.device)


            if in_train:
                # Augment waypoint prediction
                # parts of heatmap for sampling (fix offset first)
                HEATMAP_OFFSET = 5
                batch_way_heats_regional = torch.cat(
                    (waypoint_heatmap_logits[:,-HEATMAP_OFFSET:,:], 
                    waypoint_heatmap_logits[:,:-HEATMAP_OFFSET,:],
                ), dim=1)
                batch_way_heats_regional = batch_way_heats_regional.reshape(batch_size, 12, 10, 12)
                batch_sample_angle_idxes = []
                batch_sample_distance_idxes = []
                for j in range(batch_size):
                    # angle indexes with candidates
                    angle_idxes = batch_output_map[j].nonzero()[:, 0]
                    # clockwise image indexes (same as batch_x_norm)
                    img_idxes = ((angle_idxes.cpu().numpy()+5) // 10)
                    img_idxes[img_idxes==12] = 0
                    # heatmap regions for sampling
                    way_heats_regional = batch_way_heats_regional[j][img_idxes].view(img_idxes.size, -1)
                    way_heats_probs = F.softmax(way_heats_regional, 1)
                    probs_c = torch.distributions.Categorical(way_heats_probs)
                    way_heats_act = probs_c.sample().detach()
                    sample_angle_idxes = []
                    sample_distance_idxes = []
                    for k, way_act in enumerate(way_heats_act):
                        if img_idxes[k] != 0:
                            angle_pointer = (img_idxes[k] - 1) * 10 + 5
                        else:
                            angle_pointer = 0
                        sample_angle_idxes.append(torch.div(way_act, 12, rounding_mode='floor')+angle_pointer)
                        sample_distance_idxes.append(way_act%12)
                    batch_sample_angle_idxes.append(sample_angle_idxes)
                    batch_sample_distance_idxes.append(sample_distance_idxes)

            batch_angles = []
            batch_distances = []
            batch_cand_angle_fts = []
            batch_cand_idxes = []
            for j in range(batch_size):

                if in_train:
                    angle_idxes = torch.tensor(batch_sample_angle_idxes[j])
                    distance_idxes = torch.tensor(batch_sample_distance_idxes[j])
                else:
                    # angle indexes with candidates
                    angle_idxes = batch_output_map[j].nonzero()[:, 0]
                    # distance indexes for candidates
                    distance_idxes = batch_output_map[j].nonzero()[:, 1]

                # 2pi- becoz counter-clockwise is the positive direction
                angle_rad_cc = 2*math.pi-angle_idxes.float()/120*2*math.pi
                batch_angles.append(angle_rad_cc.tolist())

                batch_distances.append(
                    ((distance_idxes + 1)*0.25).tolist())
                # counter-clockwise image indexes
                img_idxes = 12 - (angle_idxes.cpu().numpy()+5) // 10
                img_idxes[img_idxes==12] = 0

                batch_cand_idxes.append([img_idxes[k] for k in range(len(img_idxes))])
                cand_angle_fts = np.concatenate([np.expand_dims(get_angle_feature(batch_angles[j][k]),0) for k in range(len(img_idxes))], axis=0)
                batch_cand_angle_fts.append(cand_angle_fts)

            view_ang_fts = get_point_angle_feature()

            ''' Extract precomputed features into variable. '''
            
            batch_view_img_fts, batch_loc_fts, batch_nav_types = [], [], []
            batch_view_lens = []
        
            for i in range(batch_size):

                view_img_fts = batch_view_fts[i]
                # cand views
                used_viewidxs = set(batch_cand_idxes[i])
                nav_types = [1] * (candidate_lengths[i]-1)
                cand_view_img_fts = view_img_fts[np.array(batch_cand_idxes[i])]
                cand_view_ang_fts = batch_cand_angle_fts[i]
              
                # non cand views
                non_cand_indxs = np.array([k for k in range(12) if k not in used_viewidxs])
                non_cand_view_img_fts = view_img_fts[non_cand_indxs]
                non_cand_view_ang_fts = view_ang_fts[non_cand_indxs]
                nav_types.extend([0] * (12 - len(used_viewidxs)))

                # combine cand views and noncand views
                view_img_fts = np.concatenate((cand_view_img_fts,non_cand_view_img_fts), 0)    # (n_views, dim_ft)
                view_ang_fts = np.concatenate((cand_view_ang_fts,non_cand_view_ang_fts), 0)

                view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
                view_loc_fts = np.concatenate([view_ang_fts, view_box_fts], 1)
            
                batch_view_img_fts.append(torch.from_numpy(view_img_fts))
                batch_loc_fts.append(torch.from_numpy(view_loc_fts))
                batch_nav_types.append(torch.LongTensor(nav_types))
                batch_view_lens.append(len(view_img_fts))

            # pad features to max_len
            batch_view_img_fts = pad_tensors(batch_view_img_fts).cuda()
            batch_loc_fts = pad_tensors(batch_loc_fts).cuda()
            batch_nav_types = pad_sequence(batch_nav_types, batch_first=True, padding_value=0).cuda()
            batch_view_lens = torch.LongTensor(batch_view_lens).cuda()



            return cand_mask, candidate_lengths, batch_angles, batch_distances, batch_view_img_fts, batch_loc_fts, batch_nav_types, batch_view_lens


        elif mode == 'navigation':

            if in_train:
                MAX_DIST = 25
                MAX_STEP = 20
            else:
                MAX_DIST = 25
                MAX_STEP = 20

            batch_size = batch_view_img_fts.shape[0]
            pano_embeds, pano_masks = self.vln_bert('panorama',
                (batch_view_img_fts, batch_loc_fts, batch_nav_types, batch_view_lens))

            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / torch.sum(pano_masks, 1, keepdim=True)
            vp_img_embeds = torch.cat([torch.zeros_like(pano_embeds[:, :1]), pano_embeds], 1)

            for i in range(batch_size):
                if len(self.traj_map[i]) == 0:
                    path_dist = 0
                else:
                    last_position = self.traj_map[i][-1][0]
                    path_dist = calc_position_distance(positions[i],last_position)
                self.traj_embeds[i].append(avg_pano_embeds[i:i+1].cpu())
                self.traj_map[i].append((positions[i],path_dist))


            batch_vp_pos_fts, batch_traj_pos_fts = [], []
            batch_traj_img_embeds = [[torch.zeros(1,768).cuda()] for i in range(batch_size)] # Stop Token
            batch_traj_step_ids = [[0] for i in range(batch_size)]
            batch_traj_lens = []
            traj_max_len = max([len(self.traj_map[i]) for i in range(batch_size)])+max(candidate_lengths)
            batch_traj_pair_dists = np.zeros((batch_size,traj_max_len,traj_max_len), dtype=np.float32)
            # dim=7 (sin(heading), cos(heading), sin(elevation), cos(elevation), line_dist, shortest_dist, shortest_step)
            for i in range(batch_size):

                ### Traj Graph dist
                traj_pair_dists = np.zeros((candidate_lengths[i]+len(self.traj_map[i]), candidate_lengths[i]+len(self.traj_map[i])), dtype=np.float32)
                for j in range(0,len(self.traj_map[i])):
                    path_dist = 0
                    for k in range(j+1,len(self.traj_map[i])):
                        path_dist += self.traj_map[i][k][1]
                        traj_pair_dists[j][k] = traj_pair_dists[k][j] = path_dist

                    for k in range(candidate_lengths[i]-1):
                        traj_pair_dists[j][len(self.traj_map[i])+k] = traj_pair_dists[len(self.traj_map[i])+k][j] = batch_distances[i][candidate_lengths[i]-2-k] + path_dist
                        for m in  range(k+1,candidate_lengths[i]-1):
                            traj_pair_dists[len(self.traj_map[i])+k][len(self.traj_map[i])+m] = traj_pair_dists[len(self.traj_map[i])+m][len(self.traj_map[i])+k] = batch_distances[i][candidate_lengths[i]-2-k] +  batch_distances[i][candidate_lengths[i]-2-m]

                traj_pair_dists = traj_pair_dists[::-1,::-1] 

                batch_traj_pair_dists[i,:traj_pair_dists.shape[0],:traj_pair_dists.shape[1]] = traj_pair_dists
                ### Global traj position embeddings
                rel_angles, rel_dists = [], []
                rel_angles.append([0,0])
                rel_dists.append([0,0,0])

                batch_traj_img_embeds[i].append(pano_embeds[i][:candidate_lengths[i]-1])
                

                for j in range(candidate_lengths[i]-1):
                    rel_heading = batch_angles[i][j]
                    rel_dist = batch_distances[i][j]
                    rel_elevation = 0.
                    rel_angles.append([rel_heading, rel_elevation])
                    rel_dists.append(
                        [rel_dist / MAX_DIST, rel_dist / MAX_DIST, 1 / MAX_STEP]
                    )
                    batch_traj_step_ids[i].append(len(self.traj_map[i])+1)
                

                path_dist = 0.
                for j in range(len(self.traj_map[i])-1,-1,-1):
                    
                    rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(self.positions[i], self.traj_map[i][j][0], base_heading=self.headings[i], base_elevation=0.)
                    rel_angles.append([rel_heading, rel_elevation])
                    rel_dists.append(
                        [rel_dist / MAX_DIST, path_dist / MAX_DIST, (self.action_step-j-1) / MAX_STEP]
                    )
                    path_dist += self.traj_map[i][j][1]
                    batch_traj_step_ids[i].append(j+1)
                    batch_traj_img_embeds[i].append(self.traj_embeds[i][j].cuda())



                rel_angles = np.array(rel_angles).astype(np.float32)
                rel_dists = np.array(rel_dists).astype(np.float32)
                rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1])
                traj_pos_fts = np.concatenate([rel_ang_fts, rel_dists], 1)

                batch_traj_step_ids[i] = torch.LongTensor(batch_traj_step_ids[i])
                batch_traj_img_embeds[i] = torch.cat(batch_traj_img_embeds[i],0)
                batch_traj_pos_fts.append(torch.from_numpy(traj_pos_fts))
                batch_traj_lens.append(batch_traj_img_embeds[i].shape[0])


                ### Local view position embeddings
                rel_angles, rel_dists = [], []
                for j in range(candidate_lengths[i]-1):
                    rel_heading = batch_angles[i][j]
                    rel_dist = batch_distances[i][j]
                    rel_elevation = 0.
                    rel_angles.append([rel_heading, rel_elevation])
                    rel_dists.append(
                        [rel_dist / MAX_DIST, rel_dist / MAX_DIST, 1 / MAX_STEP]
                    )
                rel_angles = np.array(rel_angles).astype(np.float32)
                rel_dists = np.array(rel_dists).astype(np.float32)
                rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1])
                cur_cand_pos_fts = np.concatenate([rel_ang_fts, rel_dists], 1)


                rel_angles, rel_dists = [], []
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(self.positions[i], self.start_positions[i], base_heading=self.headings[i], base_elevation=0.)
                rel_angles.append([rel_heading, rel_elevation])
                rel_dists.append(
                    [rel_dist / MAX_DIST, path_dist / MAX_DIST, self.action_step / MAX_STEP]
                )
                
                rel_angles = np.array(rel_angles).astype(np.float32)
                rel_dists = np.array(rel_dists).astype(np.float32)
                rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1])
                cur_start_pos_fts = np.concatenate([rel_ang_fts, rel_dists], 1)

 
                # add [stop] token at beginning
                vp_pos_fts = np.zeros((vp_img_embeds.size(1), 14), dtype=np.float32)
                vp_pos_fts[:, :7] = cur_start_pos_fts
                vp_pos_fts[1:len(cur_cand_pos_fts)+1, 7:] = cur_cand_pos_fts
                batch_vp_pos_fts.append(torch.from_numpy(vp_pos_fts))




            batch_traj_img_embeds = pad_tensors_wgrad(batch_traj_img_embeds).cuda()
            batch_vp_pos_fts = pad_tensors(batch_vp_pos_fts).cuda()
            batch_traj_lens = torch.LongTensor(batch_traj_lens)
            batch_traj_masks = gen_seq_masks(batch_traj_lens).cuda()
            batch_traj_step_ids = pad_sequence(batch_traj_step_ids, batch_first=True).cuda()
            batch_traj_pos_fts = pad_tensors(batch_traj_pos_fts).cuda()
            batch_traj_pair_dists = torch.tensor(batch_traj_pair_dists).cuda()
            vp_nav_masks = torch.cat([torch.ones(batch_size, 1).bool().cuda(), batch_nav_types == 1], 1)


            logits = self.vln_bert('navigation',
                (lang_feats, lang_masks, batch_traj_img_embeds, batch_traj_step_ids, batch_traj_pos_fts, batch_traj_masks, batch_traj_pair_dists, vp_img_embeds, batch_vp_pos_fts, gen_seq_masks(batch_view_lens+1), vp_nav_masks,candidate_lengths))

            for b in range(batch_size):
                logits[b,:candidate_lengths[b]] = torch.cat((logits[b,1:candidate_lengths[b]],logits[b,0:1]),0)

            return logits


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
