import numpy as np

from external.humaneva_dataset import humaneva_cameras_intrinsic_params as intrinsic_camera_params
from external.humaneva_dataset import humaneva_cameras_extrinsic_params as extrinsic_camera_params

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

def convert_cam_to_viz_dict(cam, cam_idx):
    d = {
        'orientation': cam[[11, 12, 13, 14]].numpy(),
        'translation': cam[[15, 16, 17]].numpy(),
        'res_w': cam[9].item(),
        'res_h': cam[10].item(),
        'azimuth': 0
    }
    if cam_idx == 1:
        d['azimuth'] = -90
    elif cam_idx == 2:
        d['azimuth'] = 90
    else:
        raise ValueError


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
            for i in range(len(poses_2d)): # Iterate across cameras
                out_poses_2d.append(poses_2d[i])
            if 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)): # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])
                    out_cameras.append(dataset[subject][action]['cameras'][i])
    return out_poses_3d, out_poses_2d, out_cameras


def fetch_multiview(subjects, keypoints, dataset, action_filter):
    '''Helper function to filter based on subjects and actions and collate multiple views together'''
    out_poses_3d = []
    out_poses_2d = []
    out_cameras = []

    for subject in subjects:
        subject_id = subject[-2:]
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
            out_poses_2d.append(poses_2d)
            if 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                out_poses_3d.append(poses_3d)

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
                    c = np.tile(c, (poses_3d[cam_idx].shape[0], 1))
                    camera_params.append(c)
                out_cameras.append(camera_params)
                
    return out_poses_3d, out_poses_2d, out_cameras