%% FindExtrema - Finding extrema in x direction, y direction, scale and motion space
% a 5-dimensional search as there is a fashion of 2D apparent motion between frames
% Copyright  (c) 2022 aR

function [Flag] = FindExtrema(BurstFeatureStack, Level, SpeedU, SpeedV, x, y, NumSlopeU, NumSlopeV)
    value = BurstFeatureStack(x,y,Level,SpeedU,SpeedV);
    block = BurstFeatureStack(x-1:x+1,y-1:y+1,Level-1:Level+1,1:NumSlopeU, 1:NumSlopeV);
    if (value >= 0 && value == max(block(:)))
        Flag = 1;
    elseif (value == min(block(:)))
        Flag = 1;
    else
        Flag = 0;
    end
end 