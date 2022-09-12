%% DemoBuFFVisualization - Demonstration of Burst Feature Finder for a burst with 1D apparent motion between frames %% 
%% Copyright  (c) 2022 aR

% The demo loads a burst with 1D apparent motion between frames, converts images to grayscale if required, finds features simulataneously in multi-scale muti-motion search space, and visualizes the features
% The features are color coded in a fashion to represent different 'slopes': apparent motion values. 

% Change the "Tweakables" section at the top of the script to improve results for different bursts.

clear all
clc

addpath('common/')
%% Tweakables for Burst
FilePath = './main/2D/images/';     % Burst Directory Path: If you are using your own burst, change the path here
FileFormat = '.png';                % Image Format (e.g.: '.png' '.jpg' '.tiff' 'bmp'). 
BurstLength = 5;                    % Burst Directory with 5 Images in this example. For more automatic functions: refer to utils folder

%% Load a Burst
[SingleImage, Burst] = ReadBurst(FilePath, FileFormat, BurstLength);

%% Tweakables for Feature Detection
PeakThresh = 0.006;
Octaves = 4;
Levels = 3;
FirstOctave = -1;

% We introduce a new tweakable with Burst Features: Slope (represents apparent motion).
% Slope is defined as [minimum range, maximum range, step size] in which pixel motion stack should be generated.
% Note: smaller stepsize gives better results at the expense of computation cost

SlopeSettings = [-3 3 1]; %Example way to build motion stack with apparent motion [-3 -2 -1 0 1 2 3]

BuFFPyramid = VisualizeBuFF1D(Burst, SlopeSettings, PeakThresh, Octaves, Levels);