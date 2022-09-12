%%  Histogram Smoothing: Smoothing coefficients from  5-point Gaussian  filter: 1 : 2 : 1
%% Copyright  (c) 2022 aR

function [Histogram] = HistogramSmoothing(Histogram,OrientationBins) 
    for i = 1:OrientationBins
        if (i==1)
            Previous = Histogram(OrientationBins);
            Next = Histogram(2);
        elseif (i==OrientationBins)
            Previous = Histogram(OrientationBins-1);
            Next = Histogram(1);
        else
            Previous = Histogram(i-1);
            Next = Histogram(i+1);
        end
        Histogram(i) = 0.25*Previous + 0.5*Histogram(i) + 0.25*Next;
    end
end