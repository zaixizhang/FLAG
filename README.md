# FLAG ICLR23
Molecule Generation For Target Protein Binding With Structural Motifs
<div align=center><img src="https://github.com/zaixizhang/FLAG/blob/main/flag.png" width="700"/></div>

Designing ligand molecules that bind to specific protein binding sites is a fundamental problem in structure-based drug design. Although deep generative models and geometric deep learning have made great progress in drug design, existing works either sample in the 2D graph space or fail to generate valid molecules with realistic substructures. To tackle these problems, we propose a **F**ragment-based **L**ig **A**nd **G**eneration framework (**FLAG**), to generate 3D molecules with valid and realistic substructures fragment-by-fragment. In FLAG, a motif vocabulary is constructed by extracting common molecular fragments (i.e., motif) in the dataset. At each generation step, a 3D graph neural network is first employed to encode the intermediate context information. Then, our model selects the focal motif, predicts the next motif type, and attaches the new motif. The bond lengths/angles can be quickly and accurately determined by cheminformatics tools. Finally, 
the molecular geometry is further adjusted according to the predicted rotation angle and the structure refinement.
Our model not only achieves competitive performances on conventional metrics such as binding affinity, QED, and SA, but also outperforms baselines by a
large margin in generating molecules with realistic substructures.

## 📢 News

- Please check out our latest work on structure-based drug design: *Learning Subpocket Prototypes for Generalizable Structure-based Drug Design* (ICML 2023)
  - Code: https://github.com/zaixizhang/DrugGPS_ICML23
  - Paper: https://arxiv.org/abs/2305.13997

## Install conda environment via conda yaml file
```bash
conda env create -f flag_env.yaml
conda activate flag_env
```

## Datasets
Please refer to [`README.md`](./data/README.md) in the `data` folder.

## Dataset Preprocessing and motif vocab construction

```
python build_vocab.py
```

## Training

```
python train.py
```

## Sampling
```
python motif_sample.py
```

## FLAG demo with checkpoints
Demo: https://huggingface.co/spaces/Zaixi/ICLR_FLAG

Checkpoints: https://drive.google.com/drive/folders/1NI-Tl7YzyMsfljEZXaTxbpuiO7lvUBt9?usp=drive_link

<div align=center><img src="https://github.com/zaixizhang/FLAG/blob/main/huggingface.png" width="700"/></div>

## Reference
```
@inproceedings{
zhang2023molecule,
title={Molecule Generation For Target Protein Binding with Structural Motifs},
author={ZAIXI ZHANG and Shuxin Zheng and Yaosen Min and Qi Liu},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=Rq13idF0F73}
}
```

