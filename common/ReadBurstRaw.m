%% ReadBurstRaw - Generating a burst from a single image by translating the image by a pixel %% 
%% Copyright  (c) 2022 aR

%% For further details on LFDispMousePan, LFHistEqualize, LFReadRaw, LFConvertToFloat: Check LFToolbox (Copyright (c) 2013-2020 Donald G. Dansereau)

% To change image details such as image size and format, check variables in LFReadRaw
% Note recent relase of Matlab versions allows reading raw images using rawread(image)

function [Out, Stack] = ReadBurstRaw(FilePath, FileFormat, BurstLength)

for Frame = 1: BurstLength
 Image = LFConvertToFloat(LFReadRaw([FilePath num2str(Frame) FileFormat], '16bit', [1600 1200]));
 Stack(Frame,:,:) = LFHistEqualize(Image);
end

Out = Stack(ceil(BurstLength/2),:,:);
Out = permute(Out, [2 3 1]);

%For visualisation
LFDispMousePan(Stack);