import numpy as np
import torch

from external.utils import wrap
from external.quaternion import qrot, qinverse

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

def normalize_screen_coordinates(X, w, h, pt=True): 
    assert X.shape[-1] == 2
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    if pt:
        return X/w*2 - torch.tensor([1, h/w]).to(X.device)
    else:
        return X/w*2 - [1, h/w]

    
def image_coordinates(X, w, h, pt=True):
    assert X.shape[-1] == 2
    # Reverse camera frame normalization
    if pt:
        return (X + torch.tensor([1, h/w]).to(X.device))*w/2
    else:
        return (X + [1, h/w])*w/2
    

def world_to_camera(X, R, t):
    Rt = wrap(qinverse, R) # Invert rotation
    return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - t) # Rotate and translate

    
def camera_to_world(X, R, t):
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t


def torch_world_to_camera(X, R, t):
    Rt = qinverse(R)
    return qrot(torch.tile(Rt, (*X.shape[:-1], 1)), X - t)


def torch_camera_to_world(X, R, t):
    return qrot(torch.tile(R, (*X.shape[:-1], 1)), X) + t


def camera_to_camera(X, R1, t1, R2, t2):
    world = camera_to_world(X, R1, t1)
    camera = world_to_camera(world, R2, t2)
    return camera


def torch_camera_to_camera(X, R1, t1, R2, t2):
    world = torch_camera_to_world(X, R1, t1)
    camera = torch_world_to_camera(world, R2, t2)
    return camera


def camera_3d_to_camera_2d(cam0_idx, cam0_3d, cam1_idx, cam0, cam1, fix_offset=True):
    if fix_offset:
        cam0_3d_tmp = cam0_3d.clone()
        cam0_3d_tmp[:, 1:] += cam0_3d[:, :1]
        cam0_3d = cam0_3d_tmp
    if cam0_idx == cam1_idx:
        cam1_3d = cam0_3d
    else:
        R0, t0 = cam0[0][cam_info_to_idx['orientation']], cam0[0][cam_info_to_idx['translation']]
        R1, t1 = cam1[0][cam_info_to_idx['orientation']], cam1[0][cam_info_to_idx['translation']]
        cam1_3d = torch_camera_to_camera(cam0_3d, R0, t0, R1, t1)
    cam1_intrinsic = cam1[:1, :9]
    cam1_2d_unnorm = project_to_2d(cam1_3d, cam1_intrinsic)
    w1, h1 = cam1[0][cam_info_to_idx['width']].item(), cam1[0][cam_info_to_idx['height']].item()
    cam1_2d = normalize_screen_coordinates(cam1_2d_unnorm, w=w1, h=h1)
    return cam1_2d


def project_to_2d(X, camera_params):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.
    This is a differentiable and batched reimplementation of the original MATLAB script.
    
    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]
    
    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)
        
    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    k = camera_params[..., 4:7]
    p = camera_params[..., 7:]
    
    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    r2 = torch.sum(XX[..., :2]**2, dim=len(XX.shape)-1, keepdim=True)

    radial = 1 + torch.sum(k * torch.cat((r2, r2**2, r2**3), dim=len(r2.shape)-1), dim=len(r2.shape)-1, keepdim=True)
    tan = torch.sum(p*XX, dim=len(XX.shape)-1, keepdim=True)

    XXX = XX*(radial + tan) + p*r2
    
    return f*XXX + c


def project_to_2d_linear(X, camera_params):
    """
    Project 3D points to 2D using only linear parameters (focal length and principal point).
    
    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]
    
    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)
        
    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    
    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    
    return f*XX + c