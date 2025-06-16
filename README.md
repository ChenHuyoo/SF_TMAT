# SF_TMAT
Official implementation for paper: **Teaching in adverse scenes: a statistically feedback-driven threshold and mask adjustment teacher-student framework for object detection in UAV images under adverse scenes**

The paper has been accepted by **ISPRS Journal of Photogrammetry and Remote Sensing, 2025**

![method](img/method.png)

We use [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) as the base detector. This code is built upon the original repository: https://github.com/fundamentalvision/Deformable-DETR, we thank for their excellent work.

## 1. Installation

### 1.1 Requirements

- Linux, CUDA >= 11.1, GCC >= 8.4

- Python >= 3.8

- torch >= 1.10.1, torchvision >= 0.11.2

### 1.2 Compiling Deformable DETR CUDA operators

```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```
You can download the data: [HazyDet](https://github.com/GrokCV/HazyDet), [DroneVehicle](https://github.com/VisDrone/DroneVehicle)

### 2.2 Training and evaluation
 `source_only` → `cross_domain_mae`→ `teaching`
```bash
sh configs/def-detr-base/haze/source_only.sh
sh configs/def-detr-base/haze/cross_domain_mae.sh
sh configs/def-detr-base/haze/teaching.sh
```

To evaluate the trained model and get the predicted results, run:
```bash
sh configs/def-detr-base/haze/evaluation.sh
```

## 3. model
[weights](https://drive.google.com/file/d/1auV5lf8Ydw6xvi-du34PNYSYh1dRILmr/view?usp=drive_link)



## Acknowledgement
We sincerely appreciate the authors of the following codebases which made this project possible:
- [MRT](https://github.com/JeremyZhao1998/MRT-release)  
- [HazyDet](https://github.com/GrokCV/HazyDet)

## Citation

```
@misc{chen2025teachingadversescenesstatistically,
      title={Teaching in adverse scenes: a statistically feedback-driven threshold and mask adjustment teacher-student framework for object detection in UAV images under adverse scenes}, 
      author={Hongyu Chen and Jiping Liu and Yong Wang and Jun Zhu and Dejun Feng and Yakun Xie},
      year={2025},
      eprint={2506.11175},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.11175}, 
}
```

