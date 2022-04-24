import torch

from external.camera import *
from utils import cam_info_to_idx

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    # print((predicted - target).mean())
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))


def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))


def consistency_loss(cam0_2d, cam1_2d, cam2_2d):
    '''
    Loss to ensure that views from all cameras are the same.
    cam0_2d, cam1_2d, cam2_2d are 2d projections in some (unknown) camera space (c)
    cam0_2d is the projection of 3d pose from cam0 to 2d pose in c
    '''
    assert cam0_2d.shape == cam1_2d.shape and cam1_2d.shape == cam2_2d.shape
    return (mpjpe(cam0_2d, cam1_2d) + mpjpe(cam0_2d, cam2_2d) + mpjpe(cam1_2d, cam2_2d)) / 3


def projection_loss(cam0_pred_3d, cam1_2d, cam2_2d, cam0, cam1, cam2):
    '''
    Loss after projecting cam0_pred_3d to 2d in cam1 and cam2 space
    Args:
    cam0_pred_3d      : 1 x 15 x 3
    cam1_2d, cam2_2d  : 1 x 15 x 2
    cam0, cam1, cam2  : 1 x 18
    '''
    # print(cam0_pred_3d.shape, cam1_2d.shape, cam0.shape)
    assert len(cam0_pred_3d.shape) == len(cam1_2d.shape)
    assert len(cam0.shape) == len(cam1_2d.shape) - 1
    assert cam0.shape[1] == 18
    assert (cam0_pred_3d.shape[0], cam1_2d.shape[0], cam0.shape[0]) == (1, 1, 1)
    
    R0, t0 = cam0[0][cam_info_to_idx['orientation']], cam0[0][cam_info_to_idx['translation']]
    R1, t1 = cam1[0][cam_info_to_idx['orientation']], cam1[0][cam_info_to_idx['translation']]
    R2, t2 = cam2[0][cam_info_to_idx['orientation']], cam2[0][cam_info_to_idx['translation']]
    cam0_pred_3d_tmp = cam0_pred_3d.clone()
    cam0_pred_3d_tmp[:, 1:] += cam0_pred_3d[:, :1] # resetting the offset
    cam1_pred_3d = torch_camera_to_camera(cam0_pred_3d_tmp.clone(), R0, t0, R1, t1)
    cam2_pred_3d = torch_camera_to_camera(cam0_pred_3d_tmp.clone(), R0, t0, R2, t2)
    
    cam1_intrinsic, cam2_intrinsic = cam1[:1, :9], cam2[:1, :9]
    cam1_pred_2d_unnorm = project_to_2d(cam1_pred_3d, cam1_intrinsic)
    cam2_pred_2d_unnorm = project_to_2d(cam2_pred_3d, cam2_intrinsic)
    w1, h1 = cam1[0][cam_info_to_idx['width']].item(), cam1[0][cam_info_to_idx['height']].item()
    w2, h2 = cam2[0][cam_info_to_idx['width']].item(), cam2[0][cam_info_to_idx['height']].item()
    cam1_pred_2d = normalize_screen_coordinates(cam1_pred_2d_unnorm, w=w1, h=h1)
    cam2_pred_2d = normalize_screen_coordinates(cam2_pred_2d_unnorm, w=w2, h=h2)
    
    return (mpjpe(cam1_pred_2d, cam1_2d) + mpjpe(cam2_pred_2d, cam2_2d)) / 2