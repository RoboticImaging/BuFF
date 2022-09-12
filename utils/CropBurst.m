%% CropBurst - Cropping a burst of larger size to reduce computation
%% Copyright  (c) 2022 aR

%% For further details on LFDispMousePan: Check LFToolbox (Copyright (c) 2013-2020 Donald G. Dansereau)

% Change the size of crop filters in tweakables
function [CroppedBurst] = CropBurst(Burst)

%% Tweakables
CropHeight = 10;
CropWidth = 10;

%For Visualisation
CroppedBurst = Burst(:,CropWidth:end-CropWidth+1,CropHeight:end-CropHeight+1);
LFDispMousePan(CroppedBurst)

end