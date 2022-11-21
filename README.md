# [Official] ALASCA: Adative Label Smoothing via Auxiliary Classifier
This repository contains code for AAAI-23 Paper "A Gift from Label Smoothing: Robust Training with Adaptive Label Smoothing via Auxiliary Classifier under Label Noise"

## How to use
All steps start from the root directory.

0. Set environment setup
```
pip install -r requirements.txt
```

1. ALASCA (CIFAR Symmetric 50% Noise)
```
python train_cifar.py --gpu 0 --r 0.5 --noise-type sym --epochs 120 --dataset cifar10 --loss-fn ce --wd 1e-3 --out ./saved_models/ --alasca --position_all
```

2. ALASCA + SCE (CIFAR Asymmetric 40% Noise)
```
python train_cifar.py --gpu 0 --r 0.4 --noise-type asym --epochs 120 --dataset cifar10 --loss-fn sce --wd 1e-3 --out ./saved_models/ --alasca --position_all
```

3. ALASCA + Co-teaching (CIFAR Instance-dependent 40% Noise)
```
python train_cifar.py --gpu 0 --r 0.4 --noise-type sym --epochs 120 --dataset cifar10 --loss-fn ce --num_gradual 30 --use_multi_networks --multi_networks_method coteach --out ./saved_models/ --alasca --position_all
```

## References
- Coteaching (https://github.com/bhanML/Co-teaching)
- ELR (https://github.com/shengliu66/ELR)
- Active-Passive-Losses (https://github.com/HanxunH/Active-Passive-Losses)