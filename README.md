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
