%% Descriptor Generation - Descriptor representation with thresholding
%% Copyright  (c) 2022 aR

function [Features] = DescriptorGeneration(Features,Descriptor,DescriptorMagnitudeThreshold)

Descriptor = Descriptor/norm(Descriptor);
Descriptor = min(DescriptorMagnitudeThreshold,Descriptor);
Descriptor = Descriptor/norm(Descriptor);
Features.Descriptor = Descriptor;

end
