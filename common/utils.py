import numpy as np
import torch
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

camera_idx_to_info = ['focal_length_0', 'focal_length_1', 'center_0', 'center_1', 
                 'radial_distortion_0', 'radial_distortion_1', 'radial_distortion_2', 
                 'tangential_distortion_0', 'tangential_distortion_1',
                 'width', 'height',
                 'orientation_0', 'orientation_1', 'orientation_2', 'orientation_3',
                 'translation_0', 'translation_1', 'translation_2']

cam_info_to_idx = {
    'focal_length': [0, 1],
    'center': [2, 3],
    'tangential_distortion': [4, 5, 6],
    'radial_distortion': [7, 8],
    'width': [9],
    'height': [10],
    'orientation': [11, 12, 13, 14],
    'translation': [15, 16, 17]
}
