%% FindExtrema - Finding extrema in x direction, y direction, scale and motion space
% a 4-dimensional search as there is a fashion of 1D apparent motion between frames
% Copyright  (c) 2022 aR

function [Flag] = FindExtrema(BurstFeatureStack, Level, Speed, x, y, NumSlope)
    value = BurstFeatureStack(x,y,Level,Speed);
    block = BurstFeatureStack(x-1:x+1,y-1:y+1,Level-1:Level+1,1:NumSlope);
    if (value >= 0 && value == max(block(:)))
        Flag = 1;
    elseif (value == min(block(:)))
        Flag = 1;
    else
        Flag = 0;
    end
end 