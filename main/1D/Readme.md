## Detailed Descriptoon
The algorithm for BuFF variant is designed such that motion stack is built prior to buiding multi-scale stack for feature extraction.
The design is computationally effective for higher number of burst images. This folder contains minimal functions required to perform feature extraction.
For better visualisation of results and automatic functions, refer to the [utils](/utils) folder

| Filename | Description |  
| ---------| ----------- |
| [Readme.md](/1D/Readme.md) | BuFF 1D variant: Readme file. |
| [BuFF1D](/1D/BuFF1D.m) | Function to compute BuFF features |
| [BurstKeyPointLocalization](/1D/BurstKeyPointLocalization.m) | Function to localise keypoints |
| [BurstShiftSum](/1D/BurstShiftSum.m) | Function to shift pixels and sum the shifted images into an image |
| [DemoBuFF](/1D/DemoBuFF.m) | This demonstrates feature extraction for a burst with 1D apparent motion between frames |
| [DemoBuFFVisualization](/1D/DemoBuFFVisualization.m) | This demo visualises images at various stages |
| [FeatureSelection](/1D/FeatureSelection.m) | Function for assigning feature attributes |
| [FindExtrema](/1D/FindExtrema.m) | Search space for finding extrema |
| [VisualizeBuFF1D](/1D/VisualizeBuFF1D.m) | Function to visualise various stages of features extraction|
| [VisualizeBurstShiftSum](/1D/VisualizeBurstShiftSum.m) | Function to visualise motion filter |
