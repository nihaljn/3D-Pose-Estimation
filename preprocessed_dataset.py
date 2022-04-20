import numpy as np
from torch.utils.data import Dataset

class PreprocessedDataset(Dataset):
    def __init__(self, keypoints_2d, pose_3d, cameras, keypoints_metadata, skeleton, fps, mode='frame'):
        super().__init__()
        self.keypoints_2d = keypoints_2d
        self.pose_3d = pose_3d
        self.skeleton = skeleton
        self.keypoints_metadata = keypoints_metadata
        self.fps = fps
        self.cameras = cameras
        if mode == 'frame':
            cameras = []
            for i in range(len(self.pose_3d)):
                cameras += [self.cameras[i] for _ in range(self.pose_3d[i].shape[0])]
            self.cameras = cameras
            self.keypoints_2d = np.concatenate(keypoints_2d, axis=0)
            self.pose_3d = np.concatenate(pose_3d, axis=0)
        
    def __len__(self):
        return self.pose_3d.shape[0]
    
    def __getitem__(self, index):
        return self.keypoints_2d[index], self.pose_3d[index], self.cameras[index]