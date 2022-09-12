## Design Selection
The algorithm for BuFF variant is designed such that motion stack is built prior to buiding multi-scale stack for feature extraction.
The design of building multi-scale prior to motion stack is computationally effective. This version will be released on popular request.
This folder contains minimal functions required to perform feature extraction.
For better visualisation of results and automatic functions, refer to the [utils](/utils) folder


| Filename | Description |  
| ---------| ----------- |
| [Readme.md](/2D/Readme.md) | BuFF 2D variant: Readme file. |
| [BuFF2D](/2D/BuFF2D.m) | Function to compute BuFF features |
| [BurstKeyPointLocalization](/2D/BurstKeyPointLocalization.m) | Function to localise keypoints |
| [BurstShiftSum](/2D/BurstShiftSum.m) | Function to shift pixels and sum the shifted images into an image |
| [DemoBuFF](/2D/DemoBuFF.m) | This demonstrates feature extraction for a burst with 2D apparent motion between frames |
| [DemoBuFFVisualization](/2D/DemoBuFFVisualization.m) | This demo visualises images at various stages |
| [FeatureSelection](/2D/FeatureSelection.m) | Function for assigning feature attributes |
| [FindExtrema](/2D/FindExtrema.m) | Search space for finding extrema |
| [VisualizeBuFF2D](/2D/VisualizeBuFF2D.m) | Function to visualise various stages of features extraction|
| [VisualizeBurstShiftSum](/2D/VisualizeBurstShiftSum.m) | Function to visualise motion filter |
