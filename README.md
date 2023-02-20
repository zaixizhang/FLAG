#FLAG ICLR23
This is a preliminary version of our code. We are cleaning up and will opensource the code soon.
## Install conda environment via conda yaml file
```bash
conda env create -f flag_env.yaml
conda activate flag_env
```

## Datasets
Please refer to [`README.md`](./data/README.md) in the `data` folder.

## Training

```
python train.py
```

##Reference
```@inproceedings{
zhang2023molecule,
title={Molecule Generation For Target Protein Binding with Structural Motifs},
author={ZAIXI ZHANG and Qi Liu and Shuxin Zheng and Yaosen Min},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=Rq13idF0F73}
}```

