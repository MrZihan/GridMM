# GridMM: Grid Memory Map for Vision-and-Language Navigation

### Zihan Wang and Xiangyang Li and Jiahao Yang and Yeqi Liu and Shuqiang Jiang

This repository is the official implementation of **[GridMM: Grid Memory Map for Vision-and-Language Navigation].**

>Vision-and-language navigation (VLN) enables the agent to navigate to a remote location following the natural language instruction in 3D environments. To represent the previously visited environment, most approaches for VLN implement memory using recurrent states, topological maps, or top-down semantic maps. In contrast to these approaches, we build the top-down egocentric and dynamically growing Grid Memory Map (i.e., GridMM) to structure the visited environment. From a global perspective, historical observations are projected into a unified grid map in a top-down view, which can better represent the spatial relations of the environment. From a local perspective, we further propose an instruction relevance aggregation method to capture fine-grained visual clues in each grid region. Extensive experiments are conducted on both the REVERIE, R2R, SOON datasets in the discrete environments, and the R2R-CE dataset in the continuous environments, showing the superiority of our proposed method.

![image](https://github.com/MrZihan/GridMM/blob/main/demo.gif)
The code will be released before 8/31/2023.
