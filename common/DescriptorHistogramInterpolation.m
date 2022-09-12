%% DescriptorHistogram Interpolation: Descriptor Assignment 
%% Copyright  (c) 2022 aR
 
function [Histogram] = DescriptorHistogramInterpolation(Histogram,RowBin,ColumnBin,OrientationBin,GradientMagnitudeWeight,HistogramWidth,OrientationHistogramBins)

intRow = floor(RowBin);
intColumn = floor(ColumnBin);
intOrientaiton = floor(OrientationBin);
Row = RowBin - intRow;
Column = ColumnBin - intColumn;
Orientation = OrientationBin - intOrientaiton;

for i = 0:1
    RowIndex = intRow + i;
    if (RowIndex >= 0 && RowIndex < HistogramWidth)
        for j = 0:1
            ColumnIndex = intColumn + j;
            if (ColumnIndex >=0 && ColumnIndex < HistogramWidth)
                for k = 0:1
                    OrientationIndex = mod(intOrientaiton+k,OrientationHistogramBins);
                    Update = GradientMagnitudeWeight * ( 0.5 + (Row - 0.5)*(2*i-1) ) * ( 0.5 + (Column - 0.5)*(2*j-1) ) * ( 0.5 + (Orientation - 0.5)*(2*k-1) );
                    HistogramIndex = RowIndex*HistogramWidth*OrientationHistogramBins + ColumnIndex*OrientationHistogramBins + OrientationIndex +1;
                    Histogram(HistogramIndex) = Histogram(HistogramIndex) + Update;
                end
            end
        end
    end
end

end

