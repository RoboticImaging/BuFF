%% Resizebust - Resize Burst of any length %% 
%% Copyright  (c) 2022 aR

%This is similar to the MATLAB imresize function for a single image. It uses bicubic interpolation as default
%For customized resizing of burst change tweakables: resizefactor and interpolation methods
%Note the Burst Dimension: {Burst Frames, Burst Height, Burst Width]

function [ResizedBurst] = ResizeBurst(OriginalBurst, ResizeFactor)
%Tweakables
% ResizeFactor, Interpolation: nearest, bicubic, bilinear
ResizedBurst = imresize(OriginalBurst, ResizeFactor, 'nearest'); 
end