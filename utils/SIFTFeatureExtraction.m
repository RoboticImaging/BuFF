%% SIFT feature extraction: Exisiting VL_SIFT implementation for a Single Image
%% Copyright (c) 2022 aR: This is simply using the tutorial of VL_SIFT; No modification


function [VLKeypoint, VLDescriptor] = SIFTFeatureExtraction(SingleImage)

%Tweakables
PeakThresh = PeakThresh; %colmap default:0.00667; SIFT default: 0.03
Octaves = Octaves;
Levels = Levels; 
FirstOctave = FirstOctave;

% SingleImage = im2single(imread('./2D/images/3.png'));
% To generate noisy single image: check GeneratingSyntheticSingleImage Function

[VLKeypoint,VLDescriptor] = vl_sift(im2single(SingleImage), 'PeakThresh', PeakThresh, 'Octaves', Octaves, 'Levels', Levels, 'FirstOctave', FirstOctave) ;


%Visualisation
figure(2), imshow(SingleImage)
title('SIFT Implementation', 'FontSize', 20);
hold on
for j = 1:size(VLKeypoint,2)
CurFeat = VLKeypoint(:, j);
circle( [CurFeat(1), CurFeat(2)], CurFeat(3), [], 'yellow', 'linewidth', 2 );
end