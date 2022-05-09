import numpy as np
import torch
import random

from external.human36m_dataset import h36m_cameras_intrinsic_params as intrinsic_camera_params
from external.human36m_dataset import h36m_cameras_extrinsic_params as extrinsic_camera_params


def fetch(subjects, keypoints, dataset, action_filter):
    '''Helper function to filter dataset based on subject and action filters'''
    out_poses_3d = []
    out_poses_2d = []
    out_cameras = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue
            poses_2d = keypoints[subject][action]
            for cam_idx_str in poses_2d: # Iterate across cameras
                out_poses_2d.append(poses_2d[cam_idx_str])
                
            poses_3d = dataset[subject][action]
            assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
            for i, cam_idx_str in enumerate(poses_3d): # Iterate across cameras
                out_poses_3d.append(poses_3d[cam_idx_str])
                out_cameras.append(dataset.cameras()[subject][i])
    return out_poses_3d, out_poses_2d, out_cameras


def fetch_multiview(subjects, keypoints, dataset, action_filter):
    '''Helper function to filter based on subjects and actions and collate multiple views together'''
    out_poses_3d = []
    out_poses_2d = []
    out_cameras = []

    for subject in subjects:
        subject_id = subject
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue
            poses_2d = keypoints[subject][action]
            out_poses_2d.append(list(poses_2d.values()))
            
            poses_3d = dataset[subject][action]
            assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
            out_poses_3d.append(list(poses_3d.values()))

            camera_params = []
            for cam_idx in range(3):
                in_cam = intrinsic_camera_params[cam_idx]
                ex_cam = extrinsic_camera_params[subject_id][cam_idx]
                c = np.concatenate((in_cam['focal_length'],
                                  in_cam['center'],
                                  in_cam['radial_distortion'],
                                  in_cam['tangential_distortion'],
                                  [in_cam['res_w']],
                                  [in_cam['res_h']],
                                   ex_cam['orientation'],
                                   np.asarray(ex_cam['translation'])/1000))
                c = np.tile(c, (poses_3d[f'cam_{cam_idx}'].shape[0], 1))
                camera_params.append(c)
            out_cameras.append(camera_params)
                
    return out_poses_3d, out_poses_2d, out_cameras


def convert_cam_to_viz_dict(cam, cam_idx):
    d = {
        'orientation': cam[[11, 12, 13, 14]].numpy(),
        'translation': cam[[15, 16, 17]].numpy(),
        'res_w': int(cam[9].item()),
        'res_h': int(cam[10].item())
    }
    if cam_idx == 1:
        d['azimuth'] = -90
    elif cam_idx == 2:
        d['azimuth'] = 90
    elif cam_idx == 0:
        d['azimuth'] = 0
    else:
        raise ValueError
    return d