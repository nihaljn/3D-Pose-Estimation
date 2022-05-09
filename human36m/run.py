import numpy as np
import os
import torch
import sys
import pickle
from torch.utils.data import DataLoader
import wandb

from external.camera import world_to_camera, normalize_screen_coordinates
from external.human36m_dataset import Human36mDataset
from common.model import FrameModel, SequenceModel
from common.dataset import MultiViewDataset, FrameDataset
from human36m.utils import fetch_multiview, fetch
from common.utils import set_seed
from common.loss import mpjpe


class Args:
    annotations_path = 'data/human36m/annotations.pkl'
    subjects_train = 'S1,S5,S6,S7,S8'.split(',')
    actions_train = 'Walk,Greet,Smok,Sit'.split(',')
    subjects_val = 'S11,S9'.split(',')
    actions_val = actions_train
    n_epochs = 100
    batch_size = 128
    wandb = False
    visualize_frame = True
    viz_dir = 'data/visuals/'
    model_dir = 'data/saved_models/'
    seed = 982356147
    # ckpt_path = 'data/saved_models/absurd-voice-21/last_checkpoint.pth'
    ckpt_path = None
    lr = 3e-4
    model_type = 'frame' # 'baseline' or 'frame' or 'sequence'
    
    
def run():
    args = Args()
    set_seed(args.seed)
    if args.wandb:
        wandb.init(project="vlr_project", reinit=True)
        run_name = wandb.run.name
        args.viz_dir = os.path.join(args.viz_dir, run_name)
        args.model_dir = os.path.join(args.model_dir, run_name)
        os.mkdir(args.viz_dir)
        os.mkdir(args.model_dir)
    
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
    
    criterion = None
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    scheduler = None
        
    if args.model_type == 'baseline':
        from common.train_baseline import run
        criterion = mpjpe
        poses_train_3d, poses_train_2d, cameras_train = fetch(args.subjects_train, keypoints, 
                                                              h36m_dataset, args.actions_train)
        poses_val_3d, poses_val_2d, cameras_val = fetch(args.subjects_val, keypoints, 
                                                        h36m_dataset, args.actions_val)
        train_dataset = FrameDataset(poses_train_2d, poses_train_3d, cameras_train, 
                                        keypoints_metadata, h36m_dataset.skeleton(), h36m_dataset.fps())
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True)
        val_dataset = FrameDataset(poses_val_2d, poses_val_3d, cameras_val,
                                          keypoints_metadata, h36m_dataset.skeleton(), h36m_dataset.fps())
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False)
        if args.ckpt_path != None:
            model = torch.load(args.ckpt_path).to(device)
        else:
            model = FrameModel(n_joints=17, linear_size=1024, dropout=0.5).to(device)
            
    elif args.model_type == 'frame':
        from common.train_frame import run
        poses_train_3d, poses_train_2d, cameras_train = fetch_multiview(args.subjects_train, keypoints, 
                                                                        h36m_dataset, args.actions_train)
        poses_val_3d, poses_val_2d, cameras_val = fetch_multiview(args.subjects_val, keypoints, 
                                                                  h36m_dataset, args.actions_val)
        train_dataset = MultiViewDataset(poses_train_2d, poses_train_3d, cameras_train, 
                                     keypoints_metadata, h36m_dataset.skeleton(), h36m_dataset.fps())
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        val_dataset = MultiViewDataset(poses_val_2d, poses_val_3d, cameras_val, 
                                     keypoints_metadata, h36m_dataset.skeleton(), h36m_dataset.fps())
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        if args.ckpt_path != None:
            model = torch.load(args.ckpt_path).to(device)
        else:
            model = FrameModel(n_joints=17, linear_size=1024, dropout=0.5, n_blocks=2).to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    run(args.n_epochs, train_dataloader, val_dataloader, criterion, device, model, optimizer, scheduler=scheduler, 
        use_wandb=args.wandb, visualize_frame=args.visualize_frame, 
        dataset=val_dataset, model_output_dir=args.model_dir, viz_output_dir=args.viz_dir)
    
    torch.save(model, os.path.join(args.model_dir, 'last_checkpoint.pth'))
    return
    