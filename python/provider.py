import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator

def read_burst(file_path, file_format, burst_length):
    noise_variance = 1
    stack = np.empty((0, 0, burst_length), dtype=np.float64)
    middle_frame_index = burst_length // 2

    for frame in range(burst_length):
        ax = cv2.imread(f"{file_path}{frame + 1}{file_format}", cv2.IMREAD_GRAYSCALE) / 255.0
        if frame == middle_frame_index:
            image = ax
        if stack.shape == (0, 0, burst_length):
            stack = np.empty((ax.shape[0], ax.shape[1], burst_length), dtype=np.float64)
        stack[:, :, frame] = ax + np.sqrt(noise_variance) * np.random.randn(*ax.shape)

    image_equalized = cv2.equalizeHist((image * 255).astype(np.uint8)) / 255.0
    stack_rescaled = (stack - stack.min()) / (stack.max() - stack.min())

    return image_equalized, stack_rescaled

def descriptor_generation(features, descriptor, descriptor_magnitude_threshold):
    if np.isnan(descriptor).any() or np.all(descriptor == 0):
        descriptor = np.zeros_like(descriptor)
    else:
        descriptor = descriptor / np.linalg.norm(descriptor)
        descriptor = np.minimum(descriptor_magnitude_threshold, descriptor)
        min_val = np.min(descriptor)
        max_val = np.max(descriptor)
        descriptor = 255 * (descriptor - min_val) / (max_val - min_val)

    features['Descriptor'] = descriptor

def descriptor_histogram_interpolation(histogram, row_bin, column_bin, orientation_bin, gradient_magnitude_weight, histogram_width, orientation_histogram_bins):
    int_row = int(row_bin)
    int_column = int(column_bin)
    int_orientation = int(orientation_bin)
    row = row_bin - int_row
    column = column_bin - int_column
    orientation = orientation_bin - int_orientation

    if histogram is None:
        histogram = np.zeros((histogram_width, histogram_width, orientation_histogram_bins), dtype=np.float64)

    # Handle non-scalar gradient_magnitude_weight
    if not isinstance(gradient_magnitude_weight, (int, float)):
        gradient_magnitude_weight = 0.0

    for i in range(2):
        row_index = int_row + i
        if 0 <= row_index < histogram_width:
            for j in range(2):
                column_index = int_column + j
                if 0 <= column_index < histogram_width:
                    for k in range(2):
                        orientation_index = (int_orientation + k) % orientation_histogram_bins
                        update = gradient_magnitude_weight * (0.5 + (row - 0.5) * (2 * i - 1)) * (0.5 + (column - 0.5) * (2 * j - 1)) * (0.5 + (orientation - 0.5) * (2 * k - 1))

                        histogram_index = row_index * histogram_width * orientation_histogram_bins + column_index * orientation_histogram_bins + orientation_index
                        histogram[histogram_index] += update

    return histogram

def gradient_generation(image, x, y):
    height, width = image.shape
    orientation_magnitude = np.zeros(2)

    x = int(x)
    y = int(y)

    if 1 < x < height - 1 and 1 < y < width - 1:
        dx = image[x + 1, y] - image[x - 1, y]
        dy = image[x, y + 1] - image[x, y - 1]
        orientation_magnitude[0] = np.sqrt(dx * dx + dy * dy)
        orientation_magnitude[1] = np.arctan2(dy, dx)
    else:
        # Return a consistent array with both values, even if invalid
        orientation_magnitude = np.array([-1, -1])

    return orientation_magnitude

def histogram_generation(image, x, y, orientation_bins, range_val, sigma):
    orientation_histogram = np.zeros(orientation_bins)
    smoothing_factor = 2 * sigma * sigma

    for i in range(-range_val, range_val + 1):
        for j in range(-range_val, range_val + 1):
            orientation_magnitude = gradient_generation(image, x + i, y + j)

            if orientation_magnitude[0] != -1:
                weight = np.exp(-(i * i + j * j) / smoothing_factor)
                histogram_bins = 1 + np.round(orientation_bins * (orientation_magnitude[1] + np.pi) / (2 * np.pi))
                if histogram_bins == orientation_bins + 1:
                    histogram_bins = 1
                orientation_histogram[int(histogram_bins) - 1] += weight * orientation_magnitude[0]

    return orientation_histogram

def histogram_smoothing(histogram, orientation_bins):
    smoothed_histogram = np.copy(histogram)
    for i in range(orientation_bins):
        if i == 0:
            previous = histogram[orientation_bins - 1]
            next_val = histogram[1]
        elif i == orientation_bins - 1:
            previous = histogram[orientation_bins - 2]
            next_val = histogram[0]
        else:
            previous = histogram[i - 1]
            next_val = histogram[i + 1]
        smoothed_histogram[i] = 0.25 * previous + 0.5 * histogram[i] + 0.25 * next_val

    return smoothed_histogram

def principal_curvature(buFF_image, x, y, edge_threshold):
    center = buFF_image[x, y]
    dxx = buFF_image[x, y + 1] + buFF_image[x, y - 1] - 2 * center
    dyy = buFF_image[x + 1, y] + buFF_image[x - 1, y] - 2 * center
    dxy = (buFF_image[x + 1, y + 1] + buFF_image[x - 1, y - 1] - buFF_image[x + 1, y - 1] - buFF_image[x - 1, y + 1]) / 4
    tr = dxx + dyy
    det = dxx * dyy - dxy * dxy

    if det <= 0:
        return True
    elif tr**2 / det < (edge_threshold + 1)**2 / edge_threshold:
        return False
    else:
        return True

def resize_burst(original_burst, factor):
    resized_burst = np.repeat(np.repeat(original_burst, factor, axis=0), factor, axis=1)
    return resized_burst

def LFHistEqualize(original_burst):
    normalized_burst = original_burst / np.max(original_burst)
    return normalized_burst
    
def burst_shift_sum(double_burst, tv_slope, su_slope):
    vsize, usize, tsize = double_burst.shape

    v = np.linspace(1, vsize, vsize)
    u = np.linspace(1, usize, usize)
    new_size = list(double_burst.shape)
    new_size[0:2] = [len(v), len(u)]

    v_offset_vec = np.linspace(-0.5, 0.5, tsize) * tv_slope * tsize
    u_offset_vec = np.linspace(-0.5, 0.5, tsize) * su_slope * tsize

    img_out = np.zeros(new_size, dtype=double_burst.dtype)

    for tidx in range(tsize):
        v_offset = v_offset_vec[tidx]
        u_offset = u_offset_vec[tidx]
        cur_slice = double_burst[:, :, tidx]

        interpolant = RegularGridInterpolator((v, u), cur_slice, bounds_error=False, fill_value=0)
        v_u_mesh = np.array(np.meshgrid(v + v_offset, u + u_offset, indexing='ij'))
        points = np.rollaxis(v_u_mesh, 0, 3).reshape(-1, 2)
        cur_slice = interpolant(points).reshape(len(v), len(u))

        img_out[:, :, tidx] = cur_slice

    x_image = img_out.copy()
    x_image[np.isnan(x_image)] = 0
    shifted_img = np.mean(x_image, axis=2)

    return shifted_img

def find_extrema(burst_feature_stack, level, speed_u, speed_v, x, y, num_slope_u, num_slope_v):
    value = burst_feature_stack[x, y, level, speed_u, speed_v]
    
    block = burst_feature_stack[x-1:x+2, y-1:y+2, level-1:level+2, 0:num_slope_u, 0:num_slope_v]
    
    if (value >= 0 and value == np.max(block)) or (value == np.min(block)):
        flag = 1
    else:
        flag = 0
    
    return flag

def burst_key_point_localization(buff_stack, height, width, octave, level, speed_u, speed_v, x, y, outlier_pixel, peak_threshold, count):
    num_levels = 3
    prior_smoothing = 1.6
    i = 1
    key_points = {}

    while i <= count:
        d_d = second_order_gradients(level, x, y, buff_stack)
        h = hessian(level, x, y, buff_stack)

        u, s, v = np.linalg.svd(h)
        t = np.copy(s)
        t[s != 0] = 1.0 / s[s != 0]
        inv_h = v.T @ np.diag(t) @ u.T
        update = -np.dot(inv_h, d_d)

        if np.all(np.abs(update) < 0.5):
            break

        x += round(update[0])
        y += round(update[1])
        level += round(update[2])
        speed_u += 0.01
        speed_v += 0.01

        if level < 2 or level > num_levels + 1 or x < outlier_pixel or y < outlier_pixel or x > height - outlier_pixel or y > width - outlier_pixel:
            return {}

        i += 1
        if i > count:
            return {}

    contrast = buff_stack[x, y, level] + 0.5 * np.dot(d_d.T, update)
    if np.abs(contrast) < peak_threshold / num_levels:
        return {}

    key_points['x'] = x
    key_points['y'] = y
    key_points['Octave'] = octave
    key_points['SpeedU'] = round(speed_u)
    key_points['SpeedV'] = round(speed_v)
    key_points['Level'] = level
    key_points['Offset'] = update
    key_points['ScaleOctave'] = prior_smoothing * (2 ** ((level + update[2] - 1) / num_levels))
    return key_points


def second_order_gradients(z, x, y, buff_stack):
    height, width, depth = buff_stack.shape
    if 0 <= x < height and 0 <= y < width and 0 <= z < depth:
        dx = (buff_stack[min(x + 1, height - 1), y, z] - buff_stack[max(x - 1, 0), y, z]) / 2
        dy = (buff_stack[x, min(y + 1, width - 1), z] - buff_stack[x, max(y - 1, 0), z]) / 2
        ds = (buff_stack[x, y, min(z + 1, depth - 1)] - buff_stack[x, y, max(z - 1, 0)]) / 2
        return np.array([dx, dy, ds])
    else:
        return np.array([0, 0, 0])


def hessian(z, x, y, buff_stack):
    height, width, depth = buff_stack.shape
    if 0 <= x < height and 0 <= y < width and 0 <= z < depth:
        center = buff_stack[x, y, z]
        dxx = buff_stack[min(x + 1, height - 1), y, z] + buff_stack[max(x - 1, 0), y, z] - 2 * center
        dyy = buff_stack[x, min(y + 1, width - 1), z] + buff_stack[x, max(y - 1, 0), z] - 2 * center
        dss = buff_stack[x, y, min(z + 1, depth - 1)] + buff_stack[x, y, max(z - 1, 0)] - 2 * center

        dxy = (buff_stack[min(x + 1, height - 1), min(y + 1, width - 1), z] + buff_stack[max(x - 1, 0), max(y - 1, 0), z] - buff_stack[min(x + 1, height - 1), max(y - 1, 0), z] - buff_stack[max(x - 1, 0), min(y + 1, width - 1), z]) / 4
        dxs = (buff_stack[min(x + 1, height - 1), y, min(z + 1, depth - 1)] + buff_stack[max(x - 1, 0), y, max(z - 1, 0)] - buff_stack[min(x + 1, height - 1), y, max(z - 1, 0)] - buff_stack[max(x - 1, 0), y, min(z + 1, depth - 1)]) / 4
        dys = (buff_stack[x, min(y + 1, width - 1), min(z + 1, depth - 1)] + buff_stack[x, max(y - 1, 0), max(z - 1, 0)] - buff_stack[x, max(y - 1, 0), min(z + 1, depth - 1)] - buff_stack[x, min(y + 1, width - 1), max(z - 1, 0)]) / 4

        return np.array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
    else:
        return np.zeros((3, 3))


def feature_selection(index, feature_index, keypoints, orientation_histogram, orientation_bins, orientation_peak):
    prior_smoothing = 1.6
    num_levels = 3
    features = []

    orientation_max = np.max(orientation_histogram)

    for i in range(orientation_bins):
        if i == 0:
            left = orientation_bins - 1
            right = 1
        elif i == orientation_bins - 1:
            left = orientation_bins - 2
            right = 0
        else:
            left = i - 1
            right = i + 1

        if (orientation_histogram[i] > orientation_histogram[left] and
                orientation_histogram[i] > orientation_histogram[right] and
                orientation_histogram[i] >= orientation_peak * orientation_max):

            histogram_bins = i + peak_selection(orientation_histogram[left],
                                                orientation_histogram[i],
                                                orientation_histogram[right])

            if histogram_bins - 1 <= 0:
                histogram_bins += orientation_bins

            updated_level = keypoints['Level'] + keypoints['Offset'][2]
            feature = {
                'Index': index,
                'y': (keypoints['y'] * 2 + keypoints['Offset'][1]) * 2 ** (keypoints['Octave'] - 2),
                'x': (keypoints['x'] * 2 + keypoints['Offset'][0]) * 2 ** (keypoints['Octave'] - 2),
                'Scale': prior_smoothing * 2 ** (keypoints['Octave'] - 2 + (updated_level - 1) / num_levels),
                'SlopeU': keypoints['SpeedU'],
                'SlopeV': keypoints['SpeedV'],
                'Orientation': (histogram_bins - 1) / orientation_bins * 2 * np.pi - np.pi,
                'Octave': keypoints['Octave'],
                'Level': keypoints['Level']
            }

            features.append(feature)
            feature_index += 1

    return features

def peak_selection(left, center, right):
    peak_position = 0.5 * (left - right) / (left - (2 * center + right))
    return peak_position
#%%
