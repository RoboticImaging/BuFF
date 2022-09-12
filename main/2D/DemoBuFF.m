%% DemoBuFF - Demonstration of Burst Feature Finder for a burst with 2D apparent motion between frames %% 
%% Copyright  (c) 2022 aR

% The demo loads a burst with 2D apparent motion between frames, converts images to grayscale if required, finds features simulataneously in multi-scale muti-motion search space, and visualizes the features
% The features are color coded in a fashion to represent different 'slopes': apparent motion values. 

% Change the "Tweakables" section at the top of the script to improve results for different bursts.

clear all
clc

addpath('common/')
%% Tweakables for Burst
FilePath = './main/2D/images/';     % Burst Directory Path: If you are using your own burst, change the path here
FileFormat = '.png';                % Image Format (e.g.: '.png' '.jpg' '.tiff' 'bmp'). 
BurstLength = 5;                    % Burst Directory with 5 Images in this example; For automatic functions refer to utils: eg. size(Burst,3) gives burst length if grayscale franes are stacked in third channel

%% Load a Burst
[SingleImage, Burst] = ReadBurst(FilePath, FileFormat, BurstLength);

%% Tweakables for Feature Detection
PeakThresh = 0.006;
Octaves = 4;
Levels = 3;
FirstOctave = -1;

% We introduce a new tweakable with Burst Features: Slope (represents apparent motion).
% Slope is defined as [minimum range, maximum range, step size] in which pixel motion stack should be generated.
% If the burst have only 1D apparent motion between frames, refer to BFF1D for better accuracy
% Note: smaller stepsize gives better results at the expense of computation cost

SlopeSettingsU = [-1 1 1];
SlopeSettingsV = [-1 1 1];
tic
[BurstFeature, BurstDescriptor] = BuFF2D(Burst, SlopeSettingsU, SlopeSettingsV, PeakThresh, Octaves, Levels);
toc 


%% Visualization of Burst Features
figure(1),
imshow(SingleImage)
title('BFF Implementation (Ours)', 'FontSize', 20);
hold on
MinSlope = min(BurstFeature(6,:));
SlopeRange = max(1,max(BurstFeature(6,:))-MinSlope);
for i = 1:size(BurstFeature,1)
CurFeat = BurstFeature(i,:);
Color = (CurFeat(6)-MinSlope)/SlopeRange;
RGBColor = [1-Color, 1-2*abs(Color-0.5), Color];
circle( [CurFeat(1), CurFeat(2)], CurFeat(3), [], RGBColor, 'linewidth', 2);
end


