%% BurstKeyPointLocalization: Localising keypoint position at subpixel level.
%% Fit a quadratic model to a keypoint pixel three-layered stack and update subpixel-accurate extremum.

%% Copyright  (c) 2022 aR

function [KeyPoints] = BurstKeyPointLocalization(BuFFStack, Height, Width, Octave, Level, Speed, x, y, OutlierPixel, PeakThreshold, Count)

global PriorSmoothing;
global NumLevels;

i = 1;
while (i <= Count)
    dD = SecondOrderGradients(Level,x,y);
    H = Hessian(Level,x,y);

    %method1: Singular Value Decomposition for Eigenvectors
    %[U,S,V] = svd(H); 
    %T=S;
    %T(S~=0) = 1./S(S~=0); 
    %invH = V * T' * U';
    %xa = - invH *dD;
    %xc = diag(xa);

   %method2: Sparse Equations and Least Squares
   % xa = -lsqr(H,dD); 	
   % xc = xa;

    %method3: singular value decomposition for eigenvectors
    [U,S,V] = svd(H);
    T=S;
    T(S~=0) = 1./S(S~=0); 
    invH = V * T' * U';
    Update = -invH*dD; 

    if abs(Update(1)) < 0.5 && abs(Update(2)) < 0.5 && abs(Update(3)) < 0.5
        break;
    end
    
    x = x+round(Update(1));
    y = y+round(Update(2));
    Level = Level+round(Update(3));
    Speed = Speed + 0.01;

        if (Level < 2 || Level > NumLevels+1 || x < OutlierPixel || y < OutlierPixel || x > Height-OutlierPixel || y > Width-OutlierPixel)
        KeyPoints = [];
        return;
        end

% Iterate at most 5, until the keypoint moves less than 0.5 in any direction.
i = i+1;
if (i > Count)
    KeyPoints = [];
    return;
end
end

% Contrast information is given by the ratio of eigenvalues of the 2D Hessian along the width and height of the keypoint.
    Contrast = BuFFStack(x,y,Level) + 0.5*dD'*Update;
    if (abs(Contrast) < PeakThreshold/NumLevels)
    KeyPoints = [];
    return;
    end
 
KeyPoints.x = x;
KeyPoints.y = y;
KeyPoints.Octave = Octave;
KeyPoints.Speed = round(Speed);
KeyPoints.Level = Level;
KeyPoints.Offset = Update;
KeyPoints.ScaleOctave = PriorSmoothing .* power(2,(Level+Update(3)-1)/NumLevels); 

% Second-order Central Finite Difference Approximations of the Gradients and Hessians in all dimensions.
% This is required to address discretization. 

    function [ value ] = SecondOrderGradients(z, x, y)
    dx = (BuFFStack(x+1,y,z) - BuFFStack(x-1,y,z))/2;
    dy = (BuFFStack(x,y+1,z) - BuFFStack(x,y-1,z))/2;
    ds = (BuFFStack(x,y,z+1) - BuFFStack(x,y,z-1))/2;
   
    value = [dx, dy, ds]';
end

    function [ out ] = Hessian(z, x, y)
    center = BuFFStack(x,y,z);
    dxx = BuFFStack(x+1,y,z) + BuFFStack(x-1,y,z) - 2*center;
    dyy = BuFFStack(x,y+1,z) + BuFFStack(x,y-1,z) - 2*center;
    dss = BuFFStack(x,y,z+1) + BuFFStack(x,y,z-1) - 2*center;

    dxy = (BuFFStack(x+1,y+1,z)+BuFFStack(x-1,y-1,z)-BuFFStack(x+1,y-1,z)-BuFFStack(x-1,y+1,z))/4;
    dxs = (BuFFStack(x+1,y,z+1)+BuFFStack(x-1,y,z-1)-BuFFStack(x+1,y,z-1)-BuFFStack(x-1,y,z+1))/4;
    dys = (BuFFStack(x,y+1,z+1)+BuFFStack(x,y-1,z-1)-BuFFStack(x,y-1,z+1)-BuFFStack(x,y+1,z-1))/4;

    out = [dxx,dxy,dxs;dxy,dyy,dys;dxs,dys,dss];
end

end