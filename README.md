# DynaRNA: Dynamic RNA Conformation Ensemble Generation with Diffusion Model
## Overview
DynaRNA, a diffusion-based generative model for RNA conformation ensemble. DynaRNA employs denoising diffusion probabilistic model (DDPM) with equivariant graph neural network (EGNN) to directly model RNA 3D coordinates, enabling rapid exploration of RNA conformational space. 
DynaRNA enables end-to-end generation of RNA conformation ensemble reproducing experimental geometries without the need for Multiple Sequence Alignments (MSA)  information. 

![image](https://github.com/lizxSJTU/DynaRNA/blob/main/img/DynaRNAoverview.png)

## Installation
```
git clone https://github.com/lizxSJTU/DynaRNA.git
cd DynaRNA

# Create conda environment
conda env create -f DynaRNA.yml
conda activate DynaRNA

## Install DynaRNA as a package
pip install -e .
```
After installation, you can obtain the model pkl file from 
```
https://zenodo.org/records/15600148/files/DynaRNA.pkl
```

## Inference
```
python infer_DynaRNA.py
```

## Contact
If you have any question, you can contact me with lizhengxin@sjtu.edu.cn
