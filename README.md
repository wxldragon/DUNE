# DUNE
The official implementation of our ICML 2026 paper "*Dual-branch Robust Unlearnable Examples*", by *[Xianlong Wang](https://wxldragon.github.io/), [Hangtao Zhang](https://scholar.google.com.hk/citations?user=H6wMyNEAAAAJ&hl=zh-CN), [Wenbo Pan](https://www.wenbo.io/zh-CN/), [Ziqi Zhou](https://zhou-zi7.github.io/), Changsong Jiang, Li Zeng, and [Xiaohua Jia](https://www.cs.cityu.edu.hk/~jia/).*

![ICML 2026](https://img.shields.io/badge/ICML-2026-blue.svg?style=plastic)
![Unlearnable Examples](https://img.shields.io/badge/Unlearnable-Examples-yellow.svg?style=plastic)
![Robustness](https://img.shields.io/badge/Robustness-orange.svg?style=plastic)

## Abstract
Unlearnable examples (UEs) aim to compromise model training by injecting imperceptible perturbations to clean samples. However, existing UE schemes exhibit limited robustness against advanced defenses due to their heuristic design or narrowly scoped domain perturbations. To address this, we propose DUNE, a Dual-branch UNlearnable Ensemble perturbation optimization approach. Specifically, DUNE separately optimizes perturbations in the spatial and color domains to establish the mapping between perturbations and shift-induced labels. This design extends the perturbation domain to increase noise intensity for improving robustness and drives the models to learn perturbation-oriented features with degraded generalization, thereby achieving unlearnability. To strengthen DUNE's performance, we further propose an unlearnability-enhancing ensemble strategy that aggregates diverse pre-trained models during the dual-branch optimization. Extensive experiments on benchmark datasets CIFAR-10 and ImageNet verify that DUNE's robustness outperforms 12 SOTA UE schemes under 7 mainstream defenses, yielding a lower average test accuracy of 14.95% to 50.82%.

<p align="center">
  <img src="DUNE-pipeline.png" width="700"/>
</p>

## Latest Update
| Date | Event |
|------|-------|
| **2026/05/01** | DUNE is accepted by ICML 2026. |
| **2026/05/15** | Code, README, dependencies, and released CIFAR-10 UE file are organized. |

## Start Running DUNE
- **Get code**
```shell
git clone https://github.com/wxldragon/DUNE.git
cd DUNE
git lfs pull
```

- **Build environment**
```shell
conda create -n DUNE python=3.9
conda activate DUNE
pip install -r requirements.txt
```

- **Repository layout**
```text
DUNE/
  generate_unlearnable.py      # Generate DUNE unlearnable examples
  train.py                     # Train and evaluate models on UEs
  utils_train.py               # CIFAR-10 UE dataset loader and training helpers
  madrys.py                    # PGD adversarial-training loss for defenses
  classifiers/                 # ResNet, VGG, and DenseNet backbones
  utils/                       # Data, model, and logging utilities
  ckpt/                        # Pretrained checkpoints used by UE generation
  UEs/cifar10/                 # Place released/generated CIFAR-10 UE .pkl files here
```

## Generate Unlearnable Examples
- **Prepare checkpoints**

Place the pretrained surrogate checkpoints under:
```text
ckpt/cifar10/ResNet18/
```

The generator expects checkpoint names containing `s2-e`, matching the original training checkpoint naming convention.

- **Generate CIFAR-10 DUNE UEs**
```shell
python generate_unlearnable.py --dataset cifar10 --model ResNet18 --ours --num_model 5 --alpha 0.5 --batch 1000 --num_step 30 --gpuid 0
```

Generated files are saved to:
```text
UEs/cifar10/
```

## Train and Evaluate
- **Use the released CIFAR-10 DUNE file**

Place `DUNE.pkl` under `UEs/cifar10/` before training:
```text
UEs/cifar10/DUNE.pkl
```

```shell
python train.py --ue DUNE --arch resnet18 --defense wo --gpu 0
```

- **Evaluate common defenses**
```shell
python train.py --ue DUNE --arch resnet18 --defense jpeg --gpu 0
python train.py --ue DUNE --arch resnet18 --defense gray --gpu 0
python train.py --ue DUNE --arch resnet18 --defense colorGaussian --gpu 0
python train.py --ue DUNE --arch resnet18 --defense at --gpu 0
```

Results are appended to `results.csv`.

## Acknowledge
The repository organization follows the style of [ECLIPSE](https://github.com/wxldragon/ECLIPSE). Classifier backbones are adapted from the same codebase structure for a consistent official-release layout.

## BibTeX
If you find DUNE helpful, please consider citing our paper:
```bibtex
@inproceedings{wang2026dune,
  title={Dual-branch Robust Unlearnable Examples},
  author={Wang, Xianlong and Zhang, Hangtao and Pan, Wenbo and Zhou, Ziqi and Jiang, Changsong and Zeng, Li and Jia, Xiaohua},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
  year={2026}
}
```
