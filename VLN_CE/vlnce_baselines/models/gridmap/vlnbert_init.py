import torch



def get_vlnbert_models(config=None,DATASET='R2R'):
    
    from transformers import PretrainedConfig
    from .vilmodel import GlocalTextPathNavCMT

    if DATASET=='R2R':
        model_name_or_path = 'data/pretrained_models/grid_map-models/grid_map.pt'
    elif DATASET=='RxR':
        model_name_or_path = 'data/pretrained_models/grid_map-models/grid_map_rxr.pt'

    new_ckpt_weights = {}
    if model_name_or_path is not None:
        ckpt_weights = torch.load(model_name_or_path,map_location='cpu')
        if 'vln_bert' in ckpt_weights.keys():
           ckpt_weights = ckpt_weights['vln_bert']['state_dict']
        if 'state_dict' in ckpt_weights.keys():
           ckpt_weights = ckpt_weights['state_dict']

        for k, v in ckpt_weights.items():
            if k.startswith('net.'):
                k = k[4:]
            if k.startswith('module.'):
                k = k[7:]    
            if k.startswith('bert.'):
                new_ckpt_weights[k[5:]] = v
            if k.startswith('vln_bert.'):
                new_ckpt_weights[k[9:]] = v
            else:
                new_ckpt_weights[k] = v

    if DATASET=='R2R':
        vis_config = PretrainedConfig.from_pretrained('bert-base-uncased')
    elif DATASET=='RxR':
        vis_config = PretrainedConfig.from_pretrained('data/pretrained_models/xlm')
    
    vis_config.max_action_steps = 100
    vis_config.image_feat_size = 768
    vis_config.angle_feat_size = 4

    vis_config.num_l_layers = 9
    vis_config.num_pano_layers = 2
    vis_config.num_x_layers = 4

    vis_config.fix_lang_embedding = False
    vis_config.fix_pano_embedding = False
    vis_config.fix_local_branch = False

    vis_config.update_lang_bert = not vis_config.fix_lang_embedding
    vis_config.output_attentions = True
    vis_config.pred_head_dropout_prob = 0.1
        
    visual_model = GlocalTextPathNavCMT.from_pretrained(
        pretrained_model_name_or_path=None, 
        config=vis_config, 
        state_dict=new_ckpt_weights)
     
    state_dict = torch.jit.load('data/pretrained_models/grid_map-models/ViT-B-32.pt', map_location='cpu').state_dict()
    visual_model.clip.load_state_dict(state_dict, strict=False)
        
    state_dict = torch.load('data/pretrained_models/grid_map-models/vit_base_p16_224.pth', map_location='cpu')
    visual_model.visual_encoder.load_state_dict(state_dict)

    return visual_model
