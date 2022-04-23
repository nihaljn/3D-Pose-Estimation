import numpy as np
from torch.utils.data import Dataset

class FrameDataset(Dataset):
    '''Dataset returns information for one camera and one frame at a time'''
    def __init__(self, keypoints_2d, pose_3d, cameras_intrinsics, keypoints_metadata, skeleton, fps, mode='frame'):
        super().__init__()
        self.keypoints_2d = keypoints_2d
        self.pose_3d = pose_3d
        self.skeleton = skeleton
        self.keypoints_metadata = keypoints_metadata
        self.fps = fps
        self.cameras_intrinsics = cameras_intrinsics
        if mode == 'frame':
            cameras = []
            for i in range(len(self.pose_3d)):
                cameras += [self.cameras_intrinsics[i] for _ in range(self.pose_3d[i].shape[0])]
            self.cameras = cameras
            self.keypoints_2d = np.concatenate(keypoints_2d, axis=0)
            self.pose_3d = np.concatenate(pose_3d, axis=0)
        
    def __len__(self):
        return self.pose_3d.shape[0]
    
    def __getitem__(self, index):
        return self.keypoints_2d[index], self.pose_3d[index], self.cameras[index]
    
    
class MultiViewDataset(Dataset):
    '''Dataset returns information for multiple cameras and one frame at a time'''
    def __init__(self, keypoints_2d, pose_3d, cameras, keypoints_metadata, skeleton, fps, mode='frame'):
        super().__init__()
        self.skeleton = skeleton
        self.keypoints_metadata = keypoints_metadata
        self.fps = fps
        if mode == 'frame':
            self.poses_3d = [] # list of len 3, each element is np array of poses
            self.poses_2d = [] # list of len 3, each element is np array of poses
            self.cameras = [] # list of len 3, each element is np array of poses
            for cam_idx in range(3):
                self.poses_3d.append(np.concatenate([pose[cam_idx] for pose in pose_3d], axis=0, dtype=np.float32))
                self.poses_2d.append(np.concatenate([pose[cam_idx] for pose in keypoints_2d], axis=0, dtype=np.float32))
                self.cameras.append(np.concatenate([camera[cam_idx] for camera in cameras], axis=0, dtype=np.float32))
        
    def __len__(self):
        return self.poses_3d[0].shape[0]
    
    def __getitem__(self, index):
        pose_3d = [self.poses_3d[cam_idx][index] for cam_idx in range(3)]
        pose_2d = [self.poses_2d[cam_idx][index] for cam_idx in range(3)]
        cameras = [self.cameras[cam_idx][index] for cam_idx in range(3)]
        return pose_2d, pose_3d, cameras