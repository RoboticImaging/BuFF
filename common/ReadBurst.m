%% ReadBurst - Reading a burst and converting it to grayscale images if required. %% 
%% Copyright  (c) 2022 aR

% To change the amount of noise in the image change the tweakable

function [Image, Stack] = ReadBurst(FilePath, FileFormat, BurstLength)

% Tweakables
noisevariance = 0;

Image =  im2double(imread([FilePath num2str(1) FileFormat]));
for Frame =1:BurstLength
    if size(Image,3) == 1
         Stack(:,:,Frame)  = im2double(imread([FilePath num2str(Frame) FileFormat]));
    elseif size(Image,3) == 3
        ax  = rgb2gray(im2double(imread([FilePath num2str(Frame) FileFormat])));
        Stack(:,:,Frame) = ax + sqrt(noisevariance)*randn(size(ax));
    end
end

Image = Stack(:,:,ceil(BurstLength/2));

end