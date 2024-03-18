## GridMM: Grid Memory Map for Vision-and-Language Navigation

#### Zihan Wang, Xiangyang Li, Jiahao Yang, Yeqi Liu and Shuqiang Jiang

This repository is the official implementation of **[GridMM: Grid Memory Map for Vision-and-Language Navigation](https://arxiv.org/abs/2307.12907).**

>Vision-and-language navigation (VLN) enables the agent to navigate to a remote location following the natural language instruction in 3D environments. To represent the previously visited environment, most approaches for VLN implement memory using recurrent states, topological maps, or top-down semantic maps. In contrast to these approaches, we build the top-down egocentric and dynamically growing Grid Memory Map (i.e., GridMM) to structure the visited environment. From a global perspective, historical observations are projected into a unified grid map in a top-down view, which can better represent the spatial relations of the environment. From a local perspective, we further propose an instruction relevance aggregation method to capture fine-grained visual clues in each grid region. Extensive experiments are conducted on both the REVERIE, R2R, SOON datasets in the discrete environments, and the R2R-CE dataset in the continuous environments, showing the superiority of our proposed method.

![image](https://github.com/MrZihan/GridMM/blob/main/demo.gif)


## Requirements

1. Install Matterport3D simulator for `R2R`, `REVERIE` and `SOON`: follow instructions [here](https://github.com/peteanderson80/Matterport3DSimulator).
```
export PYTHONPATH=Matterport3DSimulator/build:$PYTHONPATH
```

2. Install requirements:
```setup
conda create --name GridMM python=3.8.0
conda activate GridMM
pip install -r requirements.txt
```

3. Download annotations, preprocessed features, and trained models from [Baidu Netdisk](https://pan.baidu.com/s/1jRshMRNAhIx4VtCT0Lw1DA?pwd=beya).

4. Install Habitat simulator for `R2R-CE`: follow instructions [here](https://github.com/YicongHong/Discrete-Continuous-VLN) and [here](https://github.com/jacobkrantz/VLN-CE).


## Pretraining

Combine behavior cloning and auxiliary proxy tasks in pretraining:
```pretrain
cd pretrain_src
bash run_r2r.sh # (run_reverie.sh, run_soon.sh)
```

## Fine-tuning & Evaluation for `R2R`, `REVERIE` and `SOON`

Use pseudo interative demonstrator to fine-tune the model:
```finetune
cd map_nav_src
bash scripts/run_r2r.sh # (run_reverie.sh, run_soon.sh)
```

## Fine-tuning & Evaluation for `R2R-CE`

Use pseudo interative demonstrator to fine-tune the model:
```finetune
cd VLN_CE
bash run_GridMap.bash  # Currently, this code only supports evaluation with a single GPU.
```

## Citation

```bibtex
@inproceedings{wang2023gridmm,
  title={Gridmm: Grid memory map for vision-and-language navigation},
  author={Wang, Zihan and Li, Xiangyang and Yang, Jiahao and Liu, Yeqi and Jiang, Shuqiang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={15625--15636},
  year={2023}
}
  ```

## Acknowledgments
Our code is based on [VLN-DUET](https://github.com/cshizhe/VLN-DUET) and [CWP](https://github.com/YicongHong/Discrete-Continuous-VLN). Thanks for their great works!
