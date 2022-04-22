import numpy as np
import os
import torch
import sys
from torch.utils.data import DataLoader
import wandb

from external.camera import world_to_camera, normalize_screen_coordinates
from external.humaneva_dataset import HumanEvaDataset
from loss import mpjpe
from model import FrameModel
from run import run
from preprocessed_dataset import PreprocessedDataset

class Args:
    dataset_path = 'data/data_3d_humaneva15.npz'
    dataset_2d_path = 'data/data_2d_humaneva15_gt.npz'
    subjects_train = 'Train/S1,Train/S2,Train/S3'.split(',')
    actions_train = 'Walk,Jog,Box'.split(',')
    subjects_val = 'Validate/S1,Validate/S2,Validate/S3'.split(',')
    actions_val = actions_train
    n_epochs = 500
    batch_size = 128
    wandb = True
    visualize_frame = True
    viz_dir = 'data/visuals/'
    
    
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
    
    
def main():
    args = Args()
    if args.wandb:
        wandb.init(project="vlr_project", reinit=True)
    he_dataset = HumanEvaDataset(args.dataset_path)
    
    # convert 3D pose world coordinates to camera coordinates
    for subject in he_dataset.subjects():
        for action in he_dataset[subject].keys():
            anim = he_dataset[subject][action]
            if 'positions' in anim:
                positions_3d = []
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    pos_3d[:, 1:] -= pos_3d[:, :1] # Remove global offset, but keep trajectory in first position
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d
    
    # get 2D keypoints
    keypoints = np.load(args.dataset_2d_path, allow_pickle=True)
    keypoints_metadata = keypoints['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list(he_dataset.skeleton().joints_left()), list(he_dataset.skeleton().joints_right())
    keypoints = keypoints['positions_2d'].item() 
    
    # convert 2D pose world coordinates to screen coordinates
    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                cam = he_dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps
                
    poses_train_3d, poses_train_2d, cameras_train = fetch(args.subjects_train, keypoints, he_dataset, args.actions_train)
    poses_val_3d, poses_val_2d, cameras_val = fetch(args.subjects_val, keypoints, he_dataset, args.actions_val)
    
    train_dataset = PreprocessedDataset(poses_train_2d, poses_train_3d, cameras_train, 
                                        keypoints_metadata, he_dataset.skeleton(), he_dataset.fps())
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True)
    val_dataset = PreprocessedDataset(poses_val_2d, poses_val_3d, cameras_val,
                                      keypoints_metadata, he_dataset.skeleton(), he_dataset.fps())
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False)
    
    criterion = mpjpe
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FrameModel(n_joints=15, linear_size=1024).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    run(args.n_epochs, train_dataloader, val_dataloader, criterion, device, model, optimizer, 
        use_wandb=args.wandb, visualize_frame=args.visualize_frame, dataset=val_dataset, output_dir=args.viz_dir)
    torch.save(model, 'data/model.pth')
    return
    
    
if __name__ == '__main__':
    main()
    