%% GradientGeneration: Precomputing Gradient Magnitude and Gradient Orientations from Pixel Intensities
%% Copyright  (c) 2022 aR
 
function [OrientationMagnitude] = GradientGeneration(Image,x,y) 

[Height,Width] = size(Image);
OrientationMagnitude = [0 0];

%% Orientation Assignment
if (x > 1 && x < Height && y > 1 && y < Width)
    dx =  Image(x+1,y)- Image(x-1,y);
    dy = Image(x,y+1) - Image(x,y-1);
    OrientationMagnitude(1) = sqrt(dx*dx + dy*dy);
    OrientationMagnitude(2) = atan2(dy,dx);
else
    OrientationMagnitude = -1;
end
end
