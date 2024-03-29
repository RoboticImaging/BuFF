# BuFF: Burst Feature Finder for Light-Constrained 3D Reconstruction

We introduce burst feature finder (BuFF), **a 2D + time feature detector and descriptor that finds features with well defined scale and apparent motion within a burst of frames**. We reformulate the trajectory of a robot as multiple burst sequences and perform **3D reconstruction in low light**.

- [BuFF: Burst Feature Finder for Light-Constrained 3D Reconstruction](https://arxiv.org/pdf/2209.09470.pdf)
- submitted for oral presentation at ICRA 2023
- Authors: [Ahalya Ravendran](ahalyaravendran.com/), [Mitch Bryson](https://scholar.google.com.au/citations?user=yIFgUxwAAAAJ&hl=en/)\, and [Donald G Dansereau](https://roboticimaging.org/)
- website: [roboticimaging.org/BuFF](https://roboticimaging.org/Projects/BuFF/) with dataset details, digestable results and visualizations

Note: The code and visualisations of our ICRA2021 paper "Burst imaging for light-constrained structure-from-motion" (Burst with Merge) used in this paper for comparison is available at [roboticimaging.org/BurstSfM](https://roboticimaging.org/Projects/BurstSfM/)

<p align="center">
  <img src="https://github.com/RoboticImaging/BuFF/blob/main/assets/architecture.png" width="350" title="BuFF_architecture">
</p>

## Installation
BuFF is built with MATLAB and tested on >= R2021a versions. This repository includes code for both variations of BuFF feature extraction. We extend SIFT feature extraction to find features in a higher dimensional search space. Our implementation is inspired by SIFT implementation by [VLFeat Library](https://www.vlfeat.org/) and [LiFF implementation](https://github.com/doda42/LiFF). The functional dependencies required for evaluation and more details are discussed in 'requirements.txt':

#### Clone the Git repository.  
```bash
git clone https://github.com/RoboticImaging/BuFF/
cd BuFF/
```
## Overview
The toolkit consists of the following sub-modules.  
 - [assets](assets): Contains the source files required for creating the repository.
 - [common](common): Common scripts to run both variants of feature extraction on standard datasets. 
 - [main](main): Scripts to run two variants of the feature extractor: Burst with 1D apparent motion and Burst with 2D apparent motion.
 - [utils](utils): General utility functions for e.g. burst visualisation, histogram equalization.

## Dataset
We evaluate our feature extractor on a dataset collected in light-constrained environment using UR5e robotic arm. 
To download the complete dataset and an example seperately refer to the following links:
| Images        | Dataset |
| ------------- | ----- |
| Dataset description | [Read me](https://docs.google.com/document/d/1Ht5q7aVqLPeEca0Paon0wND1FC2mDWcwRyw0BCs2ztc/edit?usp=sharing) |
| Example burst | a burst of noisy images and corresponding ground truth with 1D and 2D apparent motion [here](https://drive.google.com/file/d/11PDClfjjMdVFbSDDxLRm28E7soqPg8FV/view?usp=sharing) (2.1GB) |
| Dataset with 1D <br> apparent motion | dataset including ground truth and noisy images [here](https://drive.google.com/file/d/19dqyBatFqHk1Yjy4QGMwWPU1Azftk9az/view?usp=sharing) (40.3GB) |
| Dataset with 2D <br> apparent motion | dataset including ground truth and noisy images [here](https://drive.google.com/file/d/1PZJmaDR7NONibRbJoyAxIZ2VrnEh9QKC/view?usp=sharing) (40.3GB) |

**Preparation:** Download the dataset from above and unpack the zip folder.
Select the directory in which images are stored and perform bias correction for accurate results.

## Update
We have now added python support for BuFF implementation in the sub-modules [python](python)
```bash
conda create -n buffenv
conda activate buffenv
pip install opencv-python numpy
cd BuFF/python/
python3 BuFF.py
```

## BibTex Citation
Please consider citing our paper if you use any of the ideas presented in the paper or code from this repository:
```
@inproceedings{ravendran2022burst,
  author    = {Ahalya Ravendran and
               Mitch Bryson and
               Donald G Dansereau},
  title     = {{BuFF: Burst Feature Finder for Light-Constrained 3D Reconstruction}},
  booktitle = {arXiv},
  year      = {2022},
}
```

## Acknowledgement
We use some functions directly from [LFToolbox](https://github.com/doda42/LFToolbox) for visualisation.
