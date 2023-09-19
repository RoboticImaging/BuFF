import os
import numpy as np
from matplotlib import pyplot as plt
from provider import *

global motion_pyramid
global motion_scale_pyramid
global BuFF_pyramid
global prior_smoothing
global num_octaves
global num_levels
global stack
global features
global num_slopesU
global num_slopesV

def BuFF2D(burst, slope_settingsU, slope_settingsV, peak_thresh, octaves, levels):
    # Preparation of Burst
    original_burst = burst
    double_burst = resize_burst(original_burst, 2)
    
    # Tweakables for Scale Space
    prior_smoothing = 1.6
    blur_init = np.sqrt(prior_smoothing**2 - 0.5**2 * 4)
    num_levels = levels
    num_octaves = octaves
    
    # Tweakables for Slope Space
    slope_setU = np.arange(slope_settingsU[0], slope_settingsU[1], slope_settingsU[2])
    num_slopesU = len(slope_setU)
    slope_setV = np.arange(slope_settingsV[0], slope_settingsV[1], slope_settingsV[2])
    num_slopesV = len(slope_setV)
    
    # Initialization of Scale Space
    scale_steps = num_levels
    mult_fact = 2**(1 / scale_steps)
    sigma = np.ones(scale_steps + 3)

    sigma[0] = prior_smoothing
    sigma[1] = prior_smoothing * np.sqrt(mult_fact * mult_fact - 1)
    
    for i in range(2, scale_steps + 3):
        sigma[i] = sigma[i - 1] * mult_fact
    
    # Initialization of Motion-Scale Pyramid
    height, width, burst_length = double_burst.shape
    motion_scale_pyramid = [None] * num_octaves
    motion_pyramid = [None] * num_octaves
    scale_motion_image = np.zeros((num_octaves, 2), dtype=int)
    scale_motion_image[0] = [height, width]
    
    for i in range(num_octaves):
        if i != 0:
            scale_motion_image[i] = [int(motion_scale_pyramid[i - 1].shape[0] / 2), int(motion_scale_pyramid[i - 1].shape[1] / 2)]
        motion_scale_pyramid[i] = np.zeros((scale_motion_image[i, 0], scale_motion_image[i, 1], scale_steps + 3, num_slopesU, num_slopesV))
        motion_pyramid[i] = np.zeros((scale_motion_image[i, 0], scale_motion_image[i, 1], num_slopesU, num_slopesV))
    
    # Building Motion Pyramid
    for slopeU in range(num_slopesU):
        for slopeV in range(num_slopesV):
            motion_image = burst_shift_sum(double_burst, slope_setV[slopeV], slope_setU[slopeU])
            motion_pyramid[0][:, :, slopeU, slopeV] = motion_image
    
    # Building Motion-Scale Pyramid
    for i in range(num_octaves):
        for j in range(scale_steps + 3):
            for slopeU in range(num_slopesU):
                for slopeV in range(num_slopesV):
                    if i == 0 and j == 0:
                        motion_scale_pyramid[i][:, :, j, slopeU, slopeV] = cv2.GaussianBlur(motion_pyramid[0][:, :, slopeU, slopeV], (0, 0), blur_init)
                    elif i != 0 and j == 0:
                        motion_scale_pyramid[i][:, :, j, slopeU, slopeV] = cv2.resize(motion_scale_pyramid[i - 1][:, :, scale_steps + 1, slopeU, slopeV], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
                    elif j != 0:
                        motion_scale_pyramid[i][:, :, j, slopeU, slopeV] = cv2.GaussianBlur(motion_scale_pyramid[i][:, :, j - 1, slopeU, slopeV], (0, 0), sigma[j])
    
    # Building BuFF Pyramid
    BuFF_pyramid = [None] * num_octaves
    for i in range(num_octaves):
        BuFF_pyramid[i] = np.zeros((scale_motion_image[i, 0], scale_motion_image[i, 1], scale_steps + 2, num_slopesU, num_slopesV))
        for j in range(scale_steps + 2):
            for slopeU in range(num_slopesU):
                for slopeV in range(num_slopesV):
                    BuFF_pyramid[i][:, :, j, slopeU, slopeV] = motion_scale_pyramid[i][:, :, j + 1, slopeU, slopeV] - motion_scale_pyramid[i][:, :, j, slopeU, slopeV]
    
    # Tweakables: Keypoint Localization
    outlier_pixel = 10
    count = 5
    edge_threshold = 10
    peak_thresh_init = 0.5 * peak_thresh / num_levels
    
    # Keypoint Localization
    stack = [{'x': 0, 'y': 0, 'Octave': 0, 'Level': 0, 'Offset': [0, 0, 0], 'ScaleOctave': 0, 'SpeedU': 0, 'SpeedV': 0}]
    index = 0
    
    for slopeU in range(num_slopesU):
        for slopeV in range(num_slopesV):
            for i in range(1, num_octaves):
                height, width = BuFF_pyramid[i][:, :, 0, 0, 0].shape
                burst_feature_stack = BuFF_pyramid[i]
                BuFF_stack = burst_feature_stack[:, :, :, slopeU, slopeV]
                
                for j in range(1, scale_steps + 1):
                    BuFF_image = burst_feature_stack[:, :, j, slopeU, slopeV]
                    
                    for x in range(outlier_pixel + 1, height - outlier_pixel):
                        for y in range(outlier_pixel + 1, width - outlier_pixel):
                            if abs(BuFF_image[x, y]) > peak_thresh_init:
                                if find_extrema(burst_feature_stack, j, slopeU, slopeV, x, y, num_slopesU, num_slopesV):
                                    burst_keypoints = burst_key_point_localization(BuFF_stack, height, width, i, j, slopeU, slopeV, x, y, outlier_pixel, peak_thresh, count)
                                    
                                    if burst_keypoints:
                                        if not principal_curvature(BuFF_image, burst_keypoints['x'], burst_keypoints['y'], edge_threshold):
                                            stack.append(burst_keypoints)
                                            index += 1
    
    # Tweakables for Orientation Assignment
    stack_length = len(stack)
    ori_sigma = 1.5
    ori_bins = 36
    ori_peak = 0.8

    features = []
    feature_index = 0

    # Orientation Assignment
    all_features = []

    # Iterate through the stack
    for e in range(stack_length):
        keypoints = stack[e]
        grad_mag = ori_sigma * keypoints['ScaleOctave']
        ori_hist = histogram_generation(motion_scale_pyramid[keypoints['Octave']][:, :, keypoints['Level'], keypoints['SpeedU'], keypoints['SpeedV']], keypoints['x'], keypoints['y'], ori_bins, int(3 * grad_mag), grad_mag)
        ori_hist = histogram_smoothing(ori_hist, ori_bins)
        features = feature_selection(e, feature_index, keypoints, ori_hist, ori_bins, ori_peak)

        # Extend the all_features list
        all_features.extend(features)

    # Tweakables for Descriptor Representation
    ori_hist_width = 4
    ori_hist_bins = 8
    desc_ori = 0
    desc_mag_thresh = 0.2
    desc_length = ori_hist_width * ori_hist_width * ori_hist_bins
    feature_length = len(all_features)

    # Descriptor Representation
    for feat_index in range(0, feature_length):
        features_set = all_features[feat_index]
        scale_motion_image = motion_scale_pyramid[features_set['Octave']][:, :, features_set['Level'], features_set['SlopeU'], features_set['SlopeV']]
        feat_width = 3 * features_set['Scale']
        radius = round(feat_width * (ori_hist_width + 1) * np.sqrt(2) / 2)
        features_ori = features_set['Orientation']
        u = features_set['x']
        v = features_set['y']
        hist_desc = np.zeros(desc_length)

        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                rot_row = j * np.cos(features_ori) - i * np.sin(features_ori)
                rot_col = j * np.sin(features_ori) + i * np.cos(features_ori)
                row_bin = rot_col / feat_width + ori_hist_width / 2 - 0.5
                col_bin = rot_row / feat_width + ori_hist_width / 2 - 0.5

                if 0 <= row_bin < ori_hist_width and 0 <= col_bin < ori_hist_width:
                    ori_mag = gradient_generation(scale_motion_image, u + i, v + j)
                    if ori_mag[0] != -1:
                        ori_mag = ori_mag[1]
                    desc_ori = desc_ori - features_ori
                    while desc_ori < 0:
                        desc_ori += 2 * np.pi
                    ori_bins = desc_ori * ori_hist_bins / (2 * np.pi)
                    weight = np.exp(-(rot_row * rot_row + rot_col * rot_col) / (2 * (0.5 * ori_hist_width * feat_width)**2))
                    if isinstance(ori_mag, (int, float)) and isinstance(weight, (int, float)):
                        weight_ori = ori_mag * weight
                    else:
                        ori_mag = np.asarray(ori_mag)
                        weight = np.asarray(weight)
                        weight_ori = ori_mag * weight
                    hist_desc = descriptor_histogram_interpolation(hist_desc, row_bin, col_bin, ori_bins, weight_ori, ori_hist_width, ori_hist_bins)

        descriptor_generation(features_set, hist_desc, desc_mag_thresh)
        all_features.append(features_set)

    # Finalizing the features extracted
    feat_scale = [Feature['Scale'] for Feature in all_features]
    feat_order = np.argsort(feat_scale)[::-1]
    descriptor = np.zeros((feature_length, desc_length))
    features_set = np.zeros((feature_length, 6))

    for i in range(feature_length):
        cur_descriptor = np.array(all_features[feat_order[i]]['Descriptor'])
        if cur_descriptor.shape[0] > 0:
            descriptor[i, :] = cur_descriptor
            features_set[i, 0] = all_features[feat_order[i]]['y']
            features_set[i, 1] = all_features[feat_order[i]]['x']
            features_set[i, 2] = all_features[feat_order[i]]['Scale']
            features_set[i, 3] = all_features[feat_order[i]]['Orientation']
            features_set[i, 4] = all_features[feat_order[i]]['SlopeU']
            features_set[i, 5] = all_features[feat_order[i]]['SlopeV']

    burst_feature = features_set
    burst_descriptor = descriptor

    return burst_feature, burst_descriptor

def demo_buff():
    file_path = '/home/ahalya/BuFF/images/'
    file_format = '.png'             
    burst_length = 5        
    
    # Load a Burst
    single_image, burst = read_burst(file_path, file_format, burst_length)
    plt.imshow(single_image)

    # Tweakables for Feature Detection
    peak_thresh = 0.005
    octaves = 4
    levels = 3
    
    # Slope settings
    slope_settings_u = np.array([-1, 0, 1])
    slope_settings_v = np.array([-1, 0, 1])

    # Run BuFF2D function
    burst_feature, burst_descriptor = BuFF2D(burst, slope_settings_u, slope_settings_v, peak_thresh, octaves, levels)
    print("Number of detected features:", burst_feature.shape[0])

    # Visualization of Burst Features
    plt.imshow(single_image, cmap='gray')
    plt.title('BuFF Implementation (Ours)')

    plt.imshow(single_image, cmap='gray')

    for feature in burst_feature:
        x, y, size, orientation = feature[0], feature[1], feature[2], feature[3]
        color = 'red'
        plt.scatter(x, y, s=size, c=color, marker='o')

    plt.show()

demo_buff()
