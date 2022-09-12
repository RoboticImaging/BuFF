%% PrincipalCurvature: Edge Rejections
% BuFF gives strong responses along edges similar to DoG stack in SIFT, these edges can be rejected using the size of principal curvature
% Reduce edge threshold <10 to remove spurious motion features at the expense of accuracy

%% Copyright  (c) 2022 aR

function [flag] = PrincipalCurvature(BuFFImage, x, y, EdgeThreshold)
Center = BuFFImage(x,y);
dxx = BuFFImage(x,y+1) + BuFFImage(x,y-1) - 2*Center; 
dyy = BuFFImage(x+1,y) + BuFFImage(x-1,y) - 2*Center; 
dxy = (BuFFImage(x+1,y+1) + BuFFImage(x-1,y-1) - BuFFImage(x+1,y-1) - BuFFImage(x-1,y+1))/4; 
Tr = dxx + dyy; 
Det = dxx * dyy - dxy * dxy; 

if ( Det <= 0 )
    flag = 1;
    return;
elseif (Tr^2 / Det < (EdgeThreshold + 1)^2 / EdgeThreshold)
    flag = 0;
else
    flag = 1;
end

end