# # TRAINING
 #flag="--exp_name cont-cwp-gridmap-ori
 #      --run-type train
 #      --exp-config run_GridMap.yaml
 #      SIMULATOR_GPU_IDS [0]
 #      TORCH_GPU_ID 0
 #      TORCH_GPU_IDS [0]
 #      IL.batch_size 2
 #      IL.lr 1e-5
 #      IL.epochs 100
 #      IL.schedule_ratio 0.50
 #      IL.decay_time 20
#       "
#python run.py $flag


 # TRAINING (Single node multiple GPUs)
 flag="--exp_name cont-cwp-gridmap-ori
       --run-type train
       --exp-config run_GridMap.yaml
       GPU_NUMBERS 4
       SIMULATOR_GPU_IDS [0,1,2,3]
       TORCH_GPU_IDS [0,1,2,3]
       IL.batch_size 2
       IL.lr 1e-5
       IL.epochs 100
       IL.schedule_ratio 0.50
       IL.decay_time 20
       "
CUDA_VISIBLE_DEVICES='4,5,6,7' python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=29503 run.py $flag


# # EVALUATION
#flag="--exp_name cont-cwp-gridmap-ori
#      --run-type eval
#      --exp-config run_GridMap.yaml
#      SIMULATOR_GPU_IDS [0,1]
#      TORCH_GPU_IDS [0,1]
#      EVAL.SPLIT val_unseen
#	  EVAL_CKPT_PATH_DIR logs/checkpoints/cont-cwp-gridmap-ori/ckpt.12.pth
#      "
#CUDA_VISIBLE_DEVICES='4,5,6,7' python3 run.py $flag

# EVAL_CKPT_PATH_DIR logs/checkpoints/cont-cwp-gridmap-ori/ckpt.12.pth  data/pretrained_models/grid_map-models/grid_map.pt  logs/checkpoints/cont-cwp-gridmap-ori/correct_best.pth

# # INFERENCE
# flag="--exp_name cont-cwp-gridmap-ori
#       --run-type inference
#       --exp-config run_GridMap.yaml
#       SIMULATOR_GPU_IDS [0]
#       TORCH_GPU_ID 0
#       TORCH_GPU_IDS [0]
#       EVAL.SAVE_RESULTS True
#       INFERENCE.PREDICTIONS_FILE test
#       INFERENCE.SPLIT test
#       INFERENCE.CKPT_PATH logs/checkpoints/cont-cwp-gridmap-ori/ckpt.12.pth
#       "
#python run.py $flag