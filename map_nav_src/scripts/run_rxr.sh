DATA_ROOT=../datasets

train_alg=dagger

features=vitbase
ft_dim=768
obj_features=vitbase
obj_ft_dim=768

ngpus=3
seed=0

name=Grid_Map-${train_alg}-${features}
name=${name}-seed.${seed}
name=${name}-init.aug.45k-new

outdir=${DATA_ROOT}/RXR/exprs_map/finetune/${name}

flag="--root_dir ${DATA_ROOT}
      --dataset r2r
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer xlm      

      --enc_full_graph
      --graph_sprels
      --fusion dynamic

      --expert_policy spl
      --train_alg ${train_alg}
      
      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2
      
      --max_action_len 20
      --max_instr_len 250

      --batch_size 2
      --lr 1e-5
      --iters 100000
      --log_every 4000
      --optim adamW

      --features ${features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --ml_weight 0.2

      --feat_dropout 0.4
      --dropout 0.5
      
      --gamma 0."

# train
CUDA_VISIBLE_DEVICES='1,2,3' python3 -m torch.distributed.launch --master_port 29501 --nproc_per_node=${ngpus} main_rxr.py $flag \
    --tokenizer bert \
	 --bert_ckpt_file ../datasets/RXR/exprs_map/pretrain/cmt-vitbase-mlm.mrc.sap-init.lxmert-aug.speaker-new/ckpts/model_step_100000.pt \
    # --eval_first

# test
#CUDA_VISIBLE_DEVICES='0' python3  main_rxr.py $flag  \
#     --tokenizer bert --test --submit \
#    --bert_ckpt_file ../datasets/RXR/exprs_map/pretrain/cmt-vitbase-mlm.mrc.sap-init.lxmert-aug.speaker-new/ckpts/model_step_95000.pt