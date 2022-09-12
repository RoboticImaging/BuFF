% Improving matching accuracy by creating keypoints with orientation

% Copyright  (c) 2022 aR

function [FeatureIndex] = FeatureSelection(Index,FeatureIndex,Keypoints,OrientationHistogram,OrientationBins,OrientationPeak)

global PriorSmoothing;
global NumLevels;
global Features;

OrientationMax = max(OrientationHistogram(:));

% For better accuracy, interpolate peak position by fitting a parabola to adjacent 3 histogram values of a keypoint .
for i = 1:OrientationBins
    if (i==1)
        Left = OrientationBins;
        Right = 2;
    elseif (i==OrientationBins)
        Left = OrientationBins-1;
        Right = 1;
    else
        Left = i-1;
        Right = i+1;
    end

    if (OrientationHistogram(i) > OrientationHistogram(Left) && OrientationHistogram(i) > OrientationHistogram(Right) && OrientationHistogram(i) >= OrientationPeak*OrientationMax )
        HistogramBins = i + PeakSelection(OrientationHistogram(Left),OrientationHistogram(i),OrientationHistogram(Right));

        if (HistogramBins -1 <= 0)
            HistogramBins = HistogramBins + OrientationBins;
        end

% Initial Image of Scale-Stack is an interpolated image of the original image with two times dimension. The KeyPoints are assigned to the original size

        UpdatedLevel = Keypoints.Level + Keypoints.Offset(3);
        Features(FeatureIndex).Index = Index;
        Features(FeatureIndex).y = [(Keypoints.y+Keypoints.Offset(2))*2^(Keypoints.Octave-2)];
        Features(FeatureIndex).x = [(Keypoints.x+Keypoints.Offset(1))*2^(Keypoints.Octave-2)];
        Features(FeatureIndex).Scale = PriorSmoothing * power(2,Keypoints.Octave-2 +(UpdatedLevel-1)/NumLevels);  
        Features(FeatureIndex).SlopeU = Keypoints.SpeedU; 
        Features(FeatureIndex).SlopeV = Keypoints.SpeedV; 
        Features(FeatureIndex).Orientation = (HistogramBins-1)/OrientationBins*2*pi - pi;
        Features(FeatureIndex).Octave = Keypoints.Octave; 
        Features(FeatureIndex).Layer = Keypoints.Level; 
        FeatureIndex = FeatureIndex + 1;
    end

end
end

function [PeakPosition] = PeakSelection(Left,Center,Right)
    PeakPosition = 0.5*(Left-Right)./(Left-(2*Center+Right));
end
