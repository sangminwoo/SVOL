# SVOL: Sketch-based Video Object Localization

![sample1](https://github.com/sangminwoo/SVOL/assets/33993419/3d6f4d30-7c05-4471-a1aa-f45265154a1d)
![sample2](https://github.com/sangminwoo/SVOL/assets/33993419/c7b9b65e-0b50-44f5-b720-b4f83c357e59)
![sample3](https://github.com/sangminwoo/SVOL/assets/33993419/9f29c7a2-e644-47b7-864a-219a64f91885)

**Paper: https://arxiv.org/abs/2304.00450**

## Getting Started
Before you begin, ensure you have the following dependencies installed on your system:

:warning: **Dependencies**
- `cuda == 10.2`
- `torch == 1.8.0`
- `torchvision == 0.9.0`
- `python == 3.8.11`
- `numpy == 1.20.3`

All experiments can be conducted with single RTX 3090 (but not limited to)


## Installation
To get started, follow these steps:

1. Clone the SVOL repository from GitHub:
```
git clone https://github.com/sangminwoo/SVOL.git
cd SVOL
```

2. Install the required Python packages listed in requirements.txt:

```
pip install -r requirements.txt
```

### Dataset Preparation
SVOL uses multiple datasets for training and evaluation. Ensure you have the following datasets ready:

- QuickDraw
- Sketchy
- TU-Berlin
- ImageNet-VID (3862/555/1861)

For the ImageNet-VID dataset, organize the data as follows:
1. In the `ILSVRC/Annotations/VID/train/` directory, move all files from `ILSVRC2015_VID_train_0000`, `ILSVRC2015_VID_train_0001`, `ILSVRC2015_VID_train_0002`, and `ILSVRC2015_VID_train_0003` to the parent directory.
2. In the `ILSVRC/Data/VID/train/` directory, move all files from `ILSVRC2015_VID_train_0000`, `ILSVRC2015_VID_train_0001`, `ILSVRC2015_VID_train_0002`, `ILSVRC2015_VID_train_0003` to the parent directory.
3. Follow [Preprocessing](https://github.com/sangminwoo/SVOL/tree/main/preprocess) steps.


### Training
You can start training the model for your selected dataset (quickdraw, sketchy, or tu-berlin) by running the respective script:
```
bash train_{dataset}.sh
```

### Evaluation
To evaluate the model, run the following command:
```
bash test.sh
```

### Configurations
For additional configuration options, refer to the `lib/configs.py` file.


## Citation
If you use SVOL in your research, please cite the following paper:
```
@article{woo2023sketch,
  title={Sketch-based Video Object Localization},
  author={Woo, Sangmin and Jeon, So-Yeong and Park, Jinyoung and Son, Minji and Lee, Sumin and Kim, Changick},
  journal={arXiv preprint arXiv:2304.00450},
  year={2023}
}
```

## Acknowledgement
We appreciate much the nicely organized codes developed by [DETR](https://github.com/facebookresearch/detr). Our codebase is built mostly based on them.