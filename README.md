## This is an code implementation of "Improving Transferability for Cross-domain Trajectory Prediction via Neural Stochastic Differential Equation", AAAI'24
Please follow below steps to run our code

## 1. Create virtual environment in Anaconda with env.yml

```
conda env create --file env.yaml -n trajsde
conda activate trajsde
```

## 2. Prepare raw dataset of nuScenes and Argoverse
Download meta data of trainval set of nuScenes from "https://www.nuscenes.org/nuscenes#download".

Download Training/Validataion/Testing dataset of motion forecasting from "https://www.argoverse.org/av1.html#download-link"

Locate them in 'data' dir as following:
```
.
├── configs
├── ...
├── data
│   ├── nuScenes
│   │   ├── maps 
│   │   ├── samples
│   │   ├── ...
│   │   └── v1.0-trainval
│   └── argodataset
│       ├── map_files
│       ├── train
│       └── val
└── train.py
```

## 3. Run preprocessing for nuScenes and Argoverse, respectively
```
mkdir preprocessed
# Argoverse
python dataset/Argoverse/Argoverse_abs.py
# nuScenes
python dataset/nuScenes/nuScenes_hivt.py
```

Then, preprocess data files are saved in 'preprocessed/Argoverse' for Argoverse and 'preprocessed/nuScenes' for nuScenes.

## 4. Make checkpoints dir and run training code
```
mkdir checkpoints
# Vanilla HiVT 
python train.py -n baseline -c configs/nusargo/hivt_nuSArgo_trmenc_mlpdec.yml
# Ours
python train.py -n nsde -c configs/nusargo/hivt_nuSArgo_sdesepenc_sdedec.yml
```
