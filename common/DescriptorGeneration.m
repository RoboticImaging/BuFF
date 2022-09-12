%% Descriptor Generation - Descriptor representation with magnitude thresholding \n
%% Copyright  (c) 2022 aR \n

function [Features] = DescriptorGeneration(Features,Descriptor,DescriptorMagnitudeThreshold)

Descriptor = Descriptor/norm(Descriptor);
Descriptor = min(DescriptorMagnitudeThreshold,Descriptor);
Descriptor = Descriptor/norm(Descriptor);
Features.Descriptor = Descriptor;

end
