%% GenerateSyntheticBurst - Generating a burst from a single image by translating the image by a pixel %% 
%% Copyright  (c) 2022 aR

% To change apparent motion in pixels, and noise, change the mentioned
% values in Tweakables section

function [Image, Stack] = GenerateSyntheticBurst(FilePath, FileFormat, BurstLength)

%% Tweakables
PixelMotion = 1;
NoiseVariance = 0;

Image = im2double(imread('./images/1.png'));
if size(Image,3) > 1
    Image = rgb2gray(im2double(imread('./images/1.png')));
end

for Frame = 1:BurstLength
    NoisyImage = Image + sqrt(NoiseVariance)*randn(size(Image));
    StackImage = imtranslate(NoisyImage,[1*(Frame-1), 1*(Frame-1)]);
    Stack(Frame,:,:) = StackImage(:,(PixelMotion*BurstLength)+1:end);
end

LFDispMousePan(Stack);