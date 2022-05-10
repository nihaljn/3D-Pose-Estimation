import numpy as np
import os
import torch
import sys
import pickle
from torch.utils.data import DataLoader
import wandb

from external.camera import *
from external.human36m_dataset import Human36mDataset
from common.model import FrameModel
from common.dataset import MultiViewDataset
from human36m.utils import *
from common.utils import *
from common.loss import *
from external.visualization import visualize


class Args:
    viz_dir = 'data/visuals/report/laced-sun-67'
    checkpoint_fp = 'data/saved_models/laced-sun-67/epoch_4.pth'
    num_samples = 5
    annotations_path = 'data/human36m/annotations.pkl'
    subjects_train = 'S1,S5,S6,S7,S8'.split(',')
    actions_train = 'Walk,Greet'.split(',')
    subjects_val = 'S11,S9'.split(',')
    actions_val = actions_train
    seed = 98356147
    
    
def visualize_frames(num_samples, dataloader, device, model, dataset, viz_output_dir):
    
    # pick an example
    cnt = 0
    for i, (pose_2d, pose_3d, cameras) in enumerate(dataloader):
        
        if i % 2 == 0:
            continue
        
        cam0_2d, cam1_2d, cam2_2d = pose_2d[0][0].to(device), pose_2d[0][1].to(device), pose_2d[0][2].to(device)
        cams = [None, None, None]
        cams[0], cams[1], cams[2] = cameras[0][0].to(device), cameras[0][1].to(device), cameras[0][2].to(device)
        n_frames = cams[0].shape[0]
        all_cam_2d = torch.cat((cam0_2d, cam1_2d, cam2_2d), dim=0)
        
        with torch.no_grad():
            cam_3d_preds = [model(cam0_2d), model(cam1_2d), model(cam2_2d)]
        
        assert dataset != None, 'Need dataset object to visualize'
        assert viz_output_dir != None, 'Need path to visualization output file'
        
        for cam_idx in range(3):
            output_fp = os.path.join(viz_output_dir, f'{cnt}_cam{cam_idx}.gif')
            data_2d = pose_2d[0][cam_idx].numpy()
            pred_3d = cam_3d_preds[cam_idx].detach().cpu().numpy()
            targ_3d = pose_3d[0][cam_idx].numpy()
            cam = convert_cam_to_viz_dict(cams[cam_idx][0], cam_idx)
            visualize(data_2d.copy(), targ_3d.copy(), pred_3d.copy(), 
                      dataset.keypoints_metadata, cam, dataset.skeleton, 
                      dataset.fps, output_fp=output_fp)
        
        cnt += 1
        if cnt == num_samples:
            return
    
    
def main():
    args = Args()
    set_seed(args.seed)
    if not os.path.isdir(args.viz_dir):
        os.mkdir(args.viz_dir)
    assert os.path.isfile(args.checkpoint_fp), 'checkpoint not found'
    
    with open(args.annotations_path, 'rb') as f:
        annotations = pickle.load(f)
    h36m_dataset = Human36mDataset(annotations['3d'])
    
    for subject in h36m_dataset.subjects():
        for action in h36m_dataset[subject].keys():
            anim = h36m_dataset[subject][action]
            for cam in anim:
                pos_3d = anim[cam] / 1000
                pos_3d[:, 1:] -= pos_3d[:, :1] # Remove global offset, but keep trajectory in first position
                anim[cam] = pos_3d.astype('float32')
    
    # convert 2D pose world coordinates to screen coordinates
    keypoints = annotations['2d']
    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for (cam_idx, (cam_idx_str, kps)) in enumerate(keypoints[subject][action].items()):
                # Normalize camera frame
                cam = h36m_dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'], pt=False)
                keypoints[subject][action][cam_idx_str] = kps.astype('float32')
    keypoints_metadata = {
        'layout_name': 'h36m',
        'num_joints': 17,
        'keypoints_symmetry': [
            [4, 5, 6, 11, 12, 13],
            [1, 2, 3, 14, 15, 16]
        ]}
                
    poses_train_3d, poses_train_2d, cameras_train = fetch_multiview(args.subjects_train, keypoints, 
                                                                        h36m_dataset, args.actions_train)
    poses_val_3d, poses_val_2d, cameras_val = fetch_multiview(args.subjects_val, keypoints, 
                                                                  h36m_dataset, args.actions_val)
    
    val_dataset = MultiViewDataset(poses_val_2d, poses_val_3d, cameras_val, 
                                 keypoints_metadata, h36m_dataset.skeleton(), h36m_dataset.fps(), mode='sequence')
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=2)
    train_dataset = MultiViewDataset(poses_train_2d, poses_train_3d, cameras_train, 
                                 keypoints_metadata, h36m_dataset.skeleton(), h36m_dataset.fps(), mode='sequence')
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(args.checkpoint_fp, map_location=device).to(device)
    model.eval()
    
    visualize_frames(args.num_samples, val_dataloader, device, model, val_dataset, viz_output_dir=args.viz_dir)
    # visualize_frames(args.num_samples, train_dataloader, device, model, train_dataset, viz_output_dir=args.viz_dir)
    # print(len(train_dataloader))
    return
    
    
if __name__ == '__main__':
    main()
    