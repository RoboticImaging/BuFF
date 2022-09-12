GeneratingSyntheticSingleImage

%% GeneratingSyntheticSingleImage - Generating a noisy single image for comparison %% 
%% Copyright  (c) 2022 aR

% To change apparent motion in pixels, and noise, change the mentioned
% values in Tweakables section

function [NoisyImage] = GeneratingSyntheticSingleImage(FilePath, FileFormat, BurstLength)

%% Tweakables
NoiseVariance = 0;

Image = im2double(imread('./images/1.png'));
if size(Image,3) > 1
    Image = rgb2gray(im2double(imread('./images/1.png')));
end

NoisyImage = Image + sqrt(NoiseVariance)*randn(size(Image));
imagesc(NoisyImage)