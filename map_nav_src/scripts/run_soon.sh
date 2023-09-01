DATA_ROOT=../datasets
train_alg=dagger

features=vitbase
ft_dim=768
obj_features=butd
obj_ft_dim=2048

ngpus=2
seed=0

name=${train_alg}-${features}
name=${name}-seed.${seed}-new

outdir=${DATA_ROOT}/SOON/exprs_map/finetune/${name}


flag="--root_dir ${DATA_ROOT}
      --dataset soon
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert

      --enc_full_graph
      --graph_sprels
      --fusion dynamic
      --multi_endpoints

      --dagger_sample sample

      --train_alg ${train_alg}
      
      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2
      
      --max_action_len 20
      --max_instr_len 100
      --max_objects 100

      --batch_size 1
      --lr 5e-5
      --iters 10000
      --log_every 100
      --optim adamW

      --features ${features}
      --obj_features ${obj_features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4
      --obj_feat_size ${obj_ft_dim}

      --ml_weight 0.2

      --feat_dropout 0.4
      --dropout 0.5
      
      --gamma 0."

CUDA_VISIBLE_DEVICES='2,3' python3 -m torch.distributed.launch \
    --nproc_per_node=${ngpus} --node_rank 0 main.py $flag  \
      --tokenizer bert \
      --bert_ckpt_file  ../datasets/SOON/exprs_map/pretrain/cmt-vitbase.butdobj-mlm.sap.og-init.lxmert-new/ckpts/model_step_22000.pt
      #--eval_first

# test
#CUDA_VISIBLE_DEVICES='2,3' python3  -m torch.distributed.launch --nproc_per_node=${ngpus} main.py $flag  \
#     --tokenizer bert \
#     --resume_file ../datasets/SOON/exprs_map/finetune/dagger-vitbase-seed.0-new/ckpts/best #\
     #--test --submit
