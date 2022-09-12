%% HistogramGeneration: Each sample added to histogram is weighted by its gradient magnitude and by a Gaussian weighted circular window with Sigma
%% Copyright  (c) 2022 aR

function [OrientationHistogram] = HistogramGeneration(Image, x, y, OrientationBins, Range, Sigma) 
 OrientationHistogram = zeros(OrientationBins,1);
SmoothingFactor = 2*Sigma*Sigma;

for i = -Range:Range
    for j = -Range:Range
        [OrientationMagnitude] = GradientGeneration(Image,x+i,y+j);
        if(OrientationMagnitude(1) ~= -1)
            Weight = exp(-(i*i+j*j)/SmoothingFactor);
            HistogramBins = 1 + round(OrientationBins*(OrientationMagnitude(2) + pi)/(2*pi));
            if(HistogramBins == OrientationBins+1)
                HistogramBins = 1;
            end
            OrientationHistogram(HistogramBins) = OrientationHistogram(HistogramBins) + Weight*OrientationMagnitude(1);
        end
    end
end
end
