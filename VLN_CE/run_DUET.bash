# # TRAINING
 #flag="--exp_name cont-cwp-duet-ori
 #      --run-type train
 #      --exp-config run_DUET.yaml
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


# # TRAINING (Single node multiple GPUs)
#flag="--exp_name cont-cwp-duet-ori
#       --run-type train
#       --exp-config run_DUET.yaml
#       GPU_NUMBERS 3
#       SIMULATOR_GPU_IDS [1,2,3]
#       TORCH_GPU_IDS [1,2,3]
#       IL.batch_size 4
#       IL.lr 1e-5
#       IL.epochs 100
#       IL.schedule_ratio 0.50
#       IL.decay_time 20
#       "
#python -m torch.distributed.launch --nproc_per_node=3 --master_port=29503 run.py $flag


# # EVALUATION
flag="--exp_name cont-cwp-duet-ori
      --run-type eval
      --exp-config run_DUET.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_ID 0
      TORCH_GPU_IDS [0]
      EVAL.SPLIT val_unseen
      EVAL_CKPT_PATH_DIR data/pretrained_models/duet-models/duet_ft.pt
      "
python3 run.py $flag
#logs/checkpoints/cont-cwp-duet-ori/ckpt.11.pth

# # INFERENCE
# flag="--exp_name cont-cwp-duet-ori
#       --run-type inference
#       --exp-config run_VLNBERT.yaml
#       SIMULATOR_GPU_IDS [0]
#       TORCH_GPU_ID 0
#       TORCH_GPU_IDS [0]
#       EVAL.SAVE_RESULTS False
#       INFERENCE.PREDICTIONS_FILE test
#       INFERENCE.SPLIT test
#       INFERENCE.CKPT_PATH logs/checkpoints/cont-cwp-vlnbert-ori/vlnbert_ckpt_best.pth
#       "
# python run.py $flag