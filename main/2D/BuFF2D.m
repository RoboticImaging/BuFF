%% Implementation of Burst Feature Finder
%% Copyright  (c) 2022 aR

function [BurstFeature,BurstDescriptor] = BuFF2D(Burst,SlopeSettingsU,SlopeSettingsV, PeakThresh, Octaves, Levels)
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
global NumSlopeU;     
global NumSlopeV;                          

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
SlopeSetU = SlopeSettingsU(1):SlopeSettingsU(3):SlopeSettingsU(2);       % Number of slopes in the defined range, If there is a memory problem with MATLAB in running the script, reduce the range of Slope or increase Step Size at the expense of accuracy
NumSlopeU = numel(SlopeSetU);    
SlopeSetV = SlopeSettingsV(1):SlopeSettingsV(3):SlopeSettingsV(2);    
NumSlopeV = numel(SlopeSetV);    


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
    MotionScalePyramid{i} = zeros(ScaleMotionImage(i,1),ScaleMotionImage(i,2),ScaleSteps+3,NumSlopeU, NumSlopeV);
    MotionPyramid{i} = zeros(ScaleMotionImage(i,1),ScaleMotionImage(i,2),NumSlopeU,NumSlopeV);
end


% -------------------------------------------------------------------------------------------- %
%% Building Motion Pyramid
% For a slope of [-2, 1, 2], there are 5 images in the pyramid each moved by pixel values of -2, -1, 0, 1 and 2 representing their apparent motion
for  islopeU = 1:NumSlopeU
     for islopeV = 1:NumSlopeV
         MotionImage = BurstShiftSum(DoubleBurst,SlopeSetV(islopeV),SlopeSetU(islopeU)); 
         %for i = 1%:NumOctaves
         %SizeScale = 1/(2.^(i-1)); 
         %MotionPyramidImage = imresize(MotionImage,SizeScale);
         MotionPyramid{1}(:,:,islopeU,islopeV) = MotionImage;
          %end
     end
end
        

% -------------------------------------------------------------------------------------------- %    
%% Building Motion-Scale Pyramid
for i = 1:NumOctaves
    for j = 1:ScaleSteps+3
         for islopeU = 1:NumSlopeU
             for islopeV = 1:NumSlopeV
                 if (i==1 && j==1)               %case 1: the first level of double image
            MotionScalePyramid{i}(:,:,j,islopeU,islopeV) = imgaussfilt(MotionPyramid{1}(:,:,islopeU,islopeV),BlurInitialization);
                 elseif (i~=1 && j==1)           %case 2: the first level of all image sizes except double image
            MotionScalePyramid{i}(:,:,j,islopeU,islopeV) = imresize(MotionScalePyramid{i-1}(:,:,ScaleSteps+1,islopeU,islopeV),0.5, 'nearest');
                 elseif(j~=1)                    %case 3: all other levels that are not one and all image sizes but not double image
            MotionScalePyramid{i}(:,:,j,islopeU,islopeV) = imgaussfilt(MotionScalePyramid{i}(:,:,j-1,islopeU,islopeV),Sigma(j));
                 end
             end
         end
    end
end


% -------------------------------------------------------------------------------------------- %
%% Building BuFF Pyramid
BuFFPyramid = cell(NumOctaves,1);
for i = 1:NumOctaves
    BuFFPyramid{i} = zeros(ScaleMotionImage(i,1),ScaleMotionImage(i,2),ScaleSteps+2,NumSlopeU,NumSlopeV);
    for j = 1:ScaleSteps+2
        for islopeU = 1:NumSlopeU
            for islopeV = 1:NumSlopeV
    BuFFPyramid{i}(:,:,j,islopeU,islopeV) = MotionScalePyramid{i}(:,:,j+1,islopeU,islopeV) - MotionScalePyramid{i}(:,:,j,islopeU,islopeV);
            end
        end
    end
end

% -------------------------------------------------------------------------------------------- %
%% Tweakables: Keypoint Localization
OutlierPixel = 10;                           % Cropping Outlier Width of an Image
Count = 5;                                   % Interpolation Count
PeakThreshold = PeakThresh;                  % Feature Contrast Threshold
EdgeThreshold = 10;                          % Principal Curvature Threshold
PeakThresholdInitialization = 0.5*PeakThreshold/NumLevels;  %Initialising Contrast Threshold depending on which level we are extracting the feature from: simple adaptive concept based on scale

%% Keypoint Localization
Stack = struct('x',0,'y',0,'Octave',0,'Level',0,'Offset',[0,0,0],'ScaleOctave',0, 'SpeedU', 0, 'SpeedV', 0);
Index = 1;

for islopeU = 1:NumSlopeU
    for islopeV = 1:NumSlopeV
        for i = 2:NumOctaves
            [Height, Width] = size(BuFFPyramid{i}(:,:,1,1,1));    
            BurstFeatureStack = BuFFPyramid{i};                      
            BuFFStack = BurstFeatureStack(:,:,:,islopeU,islopeV);

            for j = 2:ScaleSteps+1                               
                    BuFFImage = BurstFeatureStack(:,:,j,islopeU,islopeV); 

                    for x = OutlierPixel+1:Height-OutlierPixel
                        for y = OutlierPixel+1:Width-OutlierPixel

                            if(abs(BuFFImage(x,y)) > PeakThresholdInitialization)
                                if(FindExtrema(BurstFeatureStack,j,islopeU,islopeV,x,y,NumSlopeU,NumSlopeV))

                                        BurstKeyPoints = BurstKeyPointLocalization(BuFFStack,Height,Width,i,j,islopeU,islopeV,x,y,OutlierPixel,PeakThreshold,Count);

                                        if(~isempty(BurstKeyPoints))
                                            if(~PrincipalCurvature(BuFFImage,BurstKeyPoints.x,BurstKeyPoints.y,EdgeThreshold))     

                                                Stack(Index) = BurstKeyPoints; 

                                                Index = Index + 1;
                                            end
                                        end
                                
                                end
                             end
                        
                        end
                    end
            end
        end
    end
end


% -------------------------------------------------------------------------------------------- %
%% Tweakables for Orientation Assignment
StackLength = size(Stack,2);         % Stack Size
OrientationSigma = 1.5;     % Gaussian Weighted Circular Window Sigma
OrientationBins = 36;         % Orientation Histogram bins
OrientationPeak = 0.8;       % Local Peak within 80% highest peak is used in creating keypoints


% -------------------------------------------------------------------------------------------- %
%% Orientation Assignment Initialization
Features = struct('Index',0,'x',0,'y',0,'Scale',0,'Orientation',0,'SlopeU',0,'SlopeV',0,'Octave',0,'Layer',0, 'Descriptor',[]);
FeatureIndex = 1; 

% -------------------------------------------------------------------------------------------- %
%% Orientation Assignment 
for e = 1:StackLength
    KeyPoints = Stack(e);
    if (KeyPoints.x == 0) && (KeyPoints.y == 0) %if no features, ignore the building stage
        break
    end
    GradientMagnitude = OrientationSigma * KeyPoints.ScaleOctave;
    OrientationHistogram = HistogramGeneration(MotionScalePyramid{KeyPoints.Octave}(:,:,KeyPoints.Level,KeyPoints.SpeedU,KeyPoints.SpeedV),KeyPoints.x,KeyPoints.y,OrientationBins,round(3*GradientMagnitude),GradientMagnitude);
    OrientationHistogram = HistogramSmoothing(OrientationHistogram,OrientationBins);
    FeatureIndex = FeatureSelection(e,FeatureIndex,KeyPoints,OrientationHistogram,OrientationBins,OrientationPeak);
end

% -------------------------------------------------------------------------------------------- %
%%  Tweakables for Descriptor Representation
FeatureLength = size(Features,2); 
OrientationHistogramWidth = 4; 
OrientationHistogramBins = 8; 
DescriptorMagnitudeThreshold = 0.2;
DescriptorLength = OrientationHistogramWidth*OrientationHistogramWidth*OrientationHistogramBins;

% -------------------------------------------------------------------------------------------- %
%%  Initialization for Descriptor Representation
LocalFeatures = Features;
LocalStack = Stack;
LocalGaussianPyramid = MotionScalePyramid;
clear Features;
clear Stack;
clear MotionScalePyramid;
clear MotionPyramid;
clear BuFFPyramid;


% -------------------------------------------------------------------------------------------- % 
%%  Descriptor Representation

if (FeatureIndex == 0)
    BurstFeature = [0 0 0 0 0 0];
end

for FeatureIndex = 1:FeatureLength
    FeaturesSet = LocalFeatures(FeatureIndex);
    BurstKeyPoints = LocalStack(FeaturesSet.Index);
    ScaleMotionImage = LocalGaussianPyramid{BurstKeyPoints.Octave}(:,:,BurstKeyPoints.Level,BurstKeyPoints.SpeedU,BurstKeyPoints.SpeedV);
    Width = 3*BurstKeyPoints.ScaleOctave;
    Radius = round(Width*(OrientationHistogramWidth + 1)*sqrt(2)/2);
    FeatureOrientation = FeaturesSet.Orientation;
    u = BurstKeyPoints.x;
    v = BurstKeyPoints.y;
    HistogramDescriptor = zeros(1,DescriptorLength);
    
    % -------------------------------------------------------------------------------------------- % 
    %% Computing row and columns as local indices
    for i = -Radius:Radius
        for j = -Radius:Radius
            RotationRow = j*cos(FeatureOrientation) - i*sin(FeatureOrientation);
            RotationColumn = j*sin(FeatureOrientation) + i*cos(FeatureOrientation);
            RowBin = RotationColumn/Width + OrientationHistogramWidth/2 - 0.5;
            ColumnBin = RotationRow/Width + OrientationHistogramWidth/2 - 0.5;

            if (RowBin > -1 && RowBin < OrientationHistogramWidth && ColumnBin > -1 && ColumnBin < OrientationHistogramWidth)
                OrientationMagnitude = GradientGeneration(ScaleMotionImage,u+i,v+j);
                if (OrientationMagnitude(1) ~= -1)
                    DescriptorOrientation = OrientationMagnitude(2);
                    DescriptorOrientation = DescriptorOrientation - FeatureOrientation;
                    while (DescriptorOrientation < 0)
                        DescriptorOrientation = DescriptorOrientation + 2*pi;
                    end
                    OrientationBin = DescriptorOrientation * OrientationHistogramBins / (2*pi);
                    Weight = exp( -(RotationRow*RotationRow+RotationColumn*RotationColumn) / (2*(0.5*OrientationHistogramWidth*Width)^2));
                    HistogramDescriptor = DescriptorHistogramInterpolation(HistogramDescriptor,RowBin,ColumnBin,OrientationBin,OrientationMagnitude(1)*Weight,OrientationHistogramWidth,OrientationHistogramBins);
                end
            end
        end
    end
    LocalFeatures(FeatureIndex) = DescriptorGeneration(FeaturesSet,HistogramDescriptor,DescriptorMagnitudeThreshold);
end

% -------------------------------------------------------------------------------------------- % 
% Finalising the features extracted
FeatureScale = [LocalFeatures.Scale];
[~,FeatureOrder] = sort(FeatureScale,'descend');
Descriptor = zeros(FeatureLength,DescriptorLength);
FeaturesSet = zeros(FeatureLength,2);

if (FeatureIndex == 0)
BurstFeature = [0 0 0 0 0 0];
else
for i = 1:FeatureLength
    Descriptor(i,:) = LocalFeatures(FeatureOrder(i)).Descriptor;
    FeaturesSet(i,1) = LocalFeatures(FeatureOrder(i)).y;
    FeaturesSet(i,2) = LocalFeatures(FeatureOrder(i)).x;
    FeaturesSet(i,3) = LocalFeatures(FeatureOrder(i)).Scale;
    FeaturesSet(i,4) = LocalFeatures(FeatureOrder(i)).Orientation;
    FeaturesSet(i,5) = LocalFeatures(FeatureOrder(i)).SlopeU;
    FeaturesSet(i,6) = LocalFeatures(FeatureOrder(i)).SlopeV;
end
end


% -------------------------------------------------------------------------------------------- % 
TransposeKeyPoint = FeaturesSet;
TransposeDescriptor  = Descriptor;

% -------------------------------------------------------------------------------------------- % 
%Removing repeating features, especially orientation features for visualizaiton
[~,uidx] = unique(TransposeKeyPoint(:,1:3),'rows','last');
BurstFeature= TransposeKeyPoint(uidx,:);
BurstDescriptor = TransposeDescriptor(uidx,:);

end 