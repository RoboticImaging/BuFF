%% BurstShiftsum - This filter designs the motion stack by shifting and summing pixels %% 
%% Copyright  (c) 2022 aR

%% This is designed similar to LFFiltShiftSum: (Copyright 2013-2020 Donald G. Dansereau) 


% This filter take a burst of images as input, and shifts all u and v slices of an image by the given slope to compute corresponding shifted images.
% Finally, the shifted images are averaged for a single image.

%For a single slope, a burst of images generate a single output image shifted by the corresponding apparent motion.
%For a more interactive similar function and more visualisation check utils

function [ShiftedImg] = BurstShiftSum(DoubleBurst, TVSlope, SUSlope)

[vsize usize tsize] = size(DoubleBurst);

v = linspace(1,vsize, vsize);
u = linspace(1,usize, usize);
NewSize = [vsize usize tsize];
NewSize(1:2) = [length(v), length(u)];

VOffsetVec = linspace(-0.5,0.5, tsize) * TVSlope*tsize;
UOffsetVec = linspace(-0.5,0.5, tsize) * SUSlope*tsize;

ImgOut = zeros(NewSize, 'like', DoubleBurst);
for Tidx = 1:tsize
	VOffset = VOffsetVec(Tidx);
		UOffset = UOffsetVec(Tidx);
		CurSlice = squeeze(DoubleBurst(:,:,Tidx));
		
		Interpolant = griddedInterpolant( CurSlice);
		CurSlice = Interpolant( {v+VOffset, u+UOffset} );
		
		ImgOut(:,:,Tidx) = CurSlice;
    end
	if( mod(Tidx, ceil(vsize/10)) == 0 )
		fprintf('.');
	end

xImage = ImgOut;
clear ImgOut 

	W = xImage; 
	W(isnan(W)) = 0;
	xImage = W;
     
	 ShiftedImg = squeeze(mean(xImage,3)); 
end