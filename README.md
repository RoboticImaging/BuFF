# BuFF: Burst Feature Finder for Light-Constrained 3D Reconstruction

We introduce burst feature finder (BuFF), **a 2D + time feature detector and descriptor that finds features with well defined scale and apparent motion within a burst of frames**. We reformulate the trajectory of a robot as multiple burst sequences and perform **3D reconstruction in low light**.

- [BuFF: Burst Feature Finder for Light-Constrained 3D Reconstruction](https://roboticimaging.org/Papers/ravendran2022burstfeatures.pdf)
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

## Datasets
We evaluate our feature extractor on a dataset collected in light-constrained environment using UR5e robotic arm. Download an example burst of bias-corrected dataset here (aGB).
To download the original dataset seperately refer to the following links:
| Images        |       Version     | Size |
| ------------- |:-------------:| -----:|
| Bias Frames     | Ground truth | dataset (aGB) |
| Burst with 1D apparent motion    | Ground truth |  dataset (aGB) |
| Burst with 2D apparent motion | Ground truth | dataset (aGB)  |
| Bias Frames     | Conventional Noisy | dataset (aGB) |
| Burst with 1D apparent motion    | Conventional Noisy |  dataset (aGB) |
| Burst with 2D apparent motion | Conventional Noisy | dataset (aGB)  |

**Preparation:** Download the dataset from above and unpack the zip folder.
Select the directory in which images are stored and perform bias correction for accurate results.
