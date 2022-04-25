import numpy as np
import os
import torch
import sys
from torch.utils.data import DataLoader
import wandb

from external.camera import *
from external.humaneva_dataset import HumanEvaDataset
from model import FrameModel
from dataset import MultiViewDataset
from utils import *
from loss import *
from external.visualization import visualize


class Args:
    dataset_path = 'data/data_3d_humaneva15.npz'
    dataset_2d_path = 'data/data_2d_humaneva15_gt.npz'
    subjects_train = 'Train/S1,Train/S2,Train/S3'.split(',')
    actions_train = 'Walk,Jog,Box'.split(',')
    subjects_val = 'Validate/S1,Validate/S2,Validate/S3'.split(',')
    actions_val = actions_train
    viz_dir = 'data/visuals/presentation/sleek-vortex-42'
    checkpoint_fp = 'data/saved_models/sleek-vortex-42/epoch_148.pth'
    seed = 982356147
    num_samples = 5
    
    
def visualize_frames(num_samples, dataloader, device, model, dataset, viz_output_dir):
    
    # pick an example
    cnt = 0
    for pose_2d, pose_3d, cameras in dataloader:
        
        cam0_2d, cam1_2d, cam2_2d = pose_2d[0][0].to(device), pose_2d[0][1].to(device), pose_2d[0][2].to(device)
        cams = [None, None, None]
        cams[0], cams[1], cams[2] = cameras[0][0].to(device), cameras[0][1].to(device), cameras[0][2].to(device)
        n_frames = cams[0].shape[0]
        all_cam_2d = torch.cat((cam0_2d, cam1_2d, cam2_2d), dim=0)
        
        all_cam_3d_pred = model(all_cam_2d)
        # recovering per camera predictions
        cam_3d_preds = [all_cam_3d_pred[:n_frames], all_cam_3d_pred[n_frames:2*n_frames], all_cam_3d_pred[2*n_frames:]]
        
        val_losses = []
        for cam_idx in range(3):
            val_losses.append(mpjpe(cam_3d_preds[cam_idx], pose_3d[0][cam_idx]))
        val_loss = sum(val_losses) / len(val_losses)
        
        assert dataset != None, 'Need dataset object to visualize'
        assert viz_output_dir != None, 'Need path to visualization output file'
        
        for cam_idx in range(3):
            output_fp = os.path.join(viz_output_dir, f'{cnt}_cam{cam_idx}.gif')
            idx = 0
            data_2d = pose_2d[0][cam_idx].numpy()
            pred_3d = cam_3d_preds[cam_idx].detach().cpu().numpy()
            targ_3d = pose_3d[0][cam_idx].numpy()
            cam = convert_cam_to_viz_dict(cams[cam_idx][0], cam_idx)
            visualize(data_2d.copy(), targ_3d.copy(), pred_3d.copy(), 
                      dataset.keypoints_metadata, cam, dataset.skeleton, 
                      dataset.fps, output_fp=output_fp)
        
        cnt += 1
    
    
def main():
    args = Args()
    set_seed(args.seed)
    if not os.path.isdir(args.viz_dir):
        os.mkdir(args.viz_dir)
    assert os.path.isfile(args.checkpoint_fp), 'checkpoint not found'
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
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'], pt=False)
                keypoints[subject][action][cam_idx] = kps
                
    poses_train_3d, poses_train_2d, cameras_train = fetch_multiview(args.subjects_train, keypoints, he_dataset, args.actions_train)
    poses_val_3d, poses_val_2d, cameras_val = fetch_multiview(args.subjects_val, keypoints, he_dataset, args.actions_val)
    
    val_dataset = MultiViewDataset(poses_val_2d, poses_val_3d, cameras_val, 
                                 keypoints_metadata, he_dataset.skeleton(), he_dataset.fps(), mode='sequence')
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=2)
    
    criterion = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(args.checkpoint_fp).to(device)
    model.eval()
    
    visualize_frames(args.num_samples, val_dataloader, device, model, val_dataset, viz_output_dir=args.viz_dir)
    return
    
    
if __name__ == '__main__':
    main()
    