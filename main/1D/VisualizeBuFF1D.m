%% Implementation of Burst Feature Finder for Interaction with Visual Results
%% Copyright  (c) 2022 aR

function [BuFFPyramid] = VisualizeBuFF1D(Burst,SlopeSettings, PeakThresh, Octaves, Levels)
addpath('common/')

%% Initialization of Global Variables
global MotionPyramid;                      % Motion pyramid is a stack with motion-shifted imaged according to the slope values: apparent motion
global MotionScalePyramid;                 % Motion-Scale pyramid is built using each images in motion stack convolved with Gaussian filters, similar to a single image SIFT implementation 
global BuFFPyramid;                        % Similar to a DoG pyramid in SIFT, BuFFPyramid is built using the difference between images of adjacent levels in each octave of MotionScalePyramid
global PriorSmoothing;      
global NumOctaves;        
global NumLevels;          
global Stack;           
global Features;        
global NumSlope;
% -------------------------------------------------------------------------------------------- %
%% Preparation of Burst for Feature Detection
OriginalBurst = LFHistEqualize(Burst); %If your images are large in size to compute arrays, consider cropping burst 
DoubleBurst = ResizeBurst(OriginalBurst,2);   % Resizing images using nearest neighbour interpolation


% -------------------------------------------------------------------------------------------- %
%% Tweakables for Scale Space
PriorSmoothing = 1.6;                                                    % Selected based on experimental analysis for optimal repeatability according to the paper
BlurInitialization = sqrt(PriorSmoothing^2 - 0.5^2*4);                   % The paper assumes there is a blur of 0.5 in the original image. To have resulting base image with a blur of sigma, initial image is blurred with the difference
NumLevels = Levels;                                                      % Number of levels per octave
NumOctaves = Octaves;                                                    % According to the implementation: floor(log(min(size(image)))/log(2) - 2), Smaller value, smaller stack size, faster computation at the expense of bigger size features

% -------------------------------------------------------------------------------------------- %
%% Tweakables for Slope Space
SlopeSet = SlopeSettings(1):SlopeSettings(3):SlopeSettings(2);       % Number of slopes in the defined range, If there is a memory problem with MATLAB in running the script, reduce the range of Slope or increase Step Size at the expense of accuracy
NumSlope = numel(SlopeSet);        


% -------------------------------------------------------------------------------------------- %
%% Initialization of Scale Space
ScaleSteps = NumLevels;
MultiplicativeFactor = 2^(1/ScaleSteps);                                 %Difference of two nearby scales, seperated by a constant value
Sigma = ones(1,ScaleSteps+3);

Sigma(1) = PriorSmoothing;
Sigma(2) = PriorSmoothing*sqrt(MultiplicativeFactor*MultiplicativeFactor-1);
for i = 3:ScaleSteps+3
 Sigma(i) = Sigma(i-1)*MultiplicativeFactor; 
end


% -------------------------------------------------------------------------------------------- %
%% Motion-Scale Pyramid Initialization
[Height,Width,BurstLength] =  size(DoubleBurst);

MotionScalePyramid = cell(NumOctaves,1);
MotionPyramid = cell(NumOctaves,1);

ScaleMotionImage = zeros(NumOctaves,2);
ScaleMotionImage(1,:) = [Height,Width];

for i = 1:NumOctaves
    if (i~=1)
        ScaleMotionImage(i,:) = [round(size(MotionScalePyramid{i-1},1)/2),round(size(MotionScalePyramid{i-1},2)/2)];
    end
    MotionScalePyramid{i} = zeros(ScaleMotionImage(i,1),ScaleMotionImage(i,2),ScaleSteps+3,NumSlope);
    MotionPyramid{i} = zeros(ScaleMotionImage(i,1),ScaleMotionImage(i,2),NumSlope);
end


% -------------------------------------------------------------------------------------------- %
%% Building Motion Pyramid
% For a slope of [-2, 1, 2], there are 5 images in the pyramid each moved by pixel values of -2, -1, 0, 1 and 2 representing their apparent motion
for  islope = 1:NumSlope
         MotionImage = VisualizeBurstShiftSum(DoubleBurst,SlopeSet(islope),0); 
         %for i = 1%:NumOctaves
         %SizeScale = 1/(2.^(i-1)); 
         %MotionPyramidImage = imresize(MotionImage,SizeScale);
         MotionPyramid{1}(:,:,islope) = MotionImage;
          %end
end
        

% -------------------------------------------------------------------------------------------- %    
%% Building Motion-Scale Pyramid
for i = 1:NumOctaves
    for j = 1:ScaleSteps+3
         for islope = 1:NumSlope
                 if (i==1 && j==1)               %case 1: the first level of double image
            MotionScalePyramid{i}(:,:,j,islope) = imgaussfilt(MotionPyramid{1}(:,:,islope),BlurInitialization);
                 elseif (i~=1 && j==1)           %case 2: the first level of all image sizes except double image
            MotionScalePyramid{i}(:,:,j,islope) = imresize(MotionScalePyramid{i-1}(:,:,ScaleSteps+1,islope),0.5, 'nearest');
                 elseif(j~=1)                    %case 3: all other levels that are not one and all image sizes but not double image
            MotionScalePyramid{i}(:,:,j,islope) = imgaussfilt(MotionScalePyramid{i}(:,:,j-1,islope),Sigma(j));
                 end
         end
    end
end


% -------------------------------------------------------------------------------------------- %
%% Building BuFF Pyramid
BuFFPyramid = cell(NumOctaves,1);
for i = 1:NumOctaves
    BuFFPyramid{i} = zeros(ScaleMotionImage(i,1),ScaleMotionImage(i,2),ScaleSteps+2,NumSlope);
    for j = 1:ScaleSteps+2
        for islope = 1:NumSlope
    BuFFPyramid{i}(:,:,j,islope) = MotionScalePyramid{i}(:,:,j+1,islope) - MotionScalePyramid{i}(:,:,j,islope);
        end
    end
end


% -------------------------------------------------------------------------------------------- %
%% VIsualization of Intermediary Results
%% Visualization of Motion Images: 
% Intereactive Visualization
MotionStack = permute(MotionPyramid{1},[3 4 1 2]);
LFFigure(1), colormap gray, LFDispMousePan(MotionStack) %Click and drag to visualize each image in the stack; Keep the other figure tabs closed

% -------------------------------------------------------------------------------------------- %
%Visualization of all levels of a single octave 3 as a subplot figure: increase/reduce the levels based on your stack
LFFigure(2), colormap gray, subplot(2,3,1), LFDisp(DoubleBurst(:,:,1)), title('Original Image', 'FontSize', 20), subplot(2,3,2), LFDisp(MotionPyramid{1}(:,:,1)), title('Pixel Motion: -2', 'FontSize', 20), subplot(2,3,3), LFDisp(MotionPyramid{1}(:,:,2)), title('Pixel Motion: 1', 'FontSize', 20), ...
subplot(2,3,4), LFDisp(MotionPyramid{1}(:,:,3)), title('Pixel Motion: 0', 'FontSize', 20), subplot(2,3,5), LFDisp(MotionPyramid{1}(:,:,4)), title('Pixel Motion: 1', 'FontSize', 20), subplot(2,3,6), LFDisp(MotionPyramid{1}(:,:,5)), title('Pixel Motion: 2', 'FontSize', 20)

% -------------------------------------------------------------------------------------------- %
%% Visualization of MotionScalePyramid Images:
LFFigure(2), colormap gray, subplot(2,3,1), LFDisp(MotionScalePyramid{3}(:,:,1,1)), title('Level 1', 'FontSize', 20), subplot(2,3,2), LFDisp(MotionScalePyramid{3}(:,:,1,2)), title('Level 2', 'FontSize', 20), subplot(2,3,3), LFDisp(MotionScalePyramid{3}(:,:,1,3)), title('Level 3', 'FontSize', 20), ...
subplot(2,3,4), LFDisp(MotionScalePyramid{3}(:,:,1,4)), title('Level 4', 'FontSize', 20), subplot(2,3,5), LFDisp(MotionScalePyramid{3}(:,:,1,5)), title('Level 5', 'FontSize', 20), subplot(2,3,6), LFDisp(MotionScalePyramid{3}(:,:,1,6)), title('Level 6', 'FontSize', 20)
                    
% -------------------------------------------------------------------------------------------- %
%% Visualization of BuFF Images:      
LFFigure(2), colormap gray, subplot(2,3,1), LFDisp(DoubleBurst(:,:,1)), title('Original Image', 'FontSize', 20), subplot(2,3,2), LFDisp(BuFFPyramid{3}(:,:,1,1)), title('Level 2 - Level1', 'FontSize', 20), subplot(2,3,3), LFDisp(BuFFPyramid{3}(:,:,1,2)), title('Level 3 - Level 2', 'FontSize', 20), ...
subplot(2,3,4), LFDisp(BuFFPyramid{3}(:,:,1,3)), title('Level 4 - Level 3', 'FontSize', 20), subplot(2,3,5), LFDisp(BuFFPyramid{3}(:,:,1,4)), title('Level 5 - Level 4', 'FontSize', 20), subplot(2,3,6), LFDisp(BuFFPyramid{3}(:,:,1,5)), title('Level 6 - Level 5', 'FontSize', 20);
