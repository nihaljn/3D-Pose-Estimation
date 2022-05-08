import numpy as np
import os
import torch
import sys
from torch.utils.data import DataLoader
import wandb

from external.camera import world_to_camera, normalize_screen_coordinates
from external.humaneva_dataset import HumanEvaDataset
from common.model import FrameModel, SequenceModel
from humaneva.dataset import MultiViewDataset, FrameDataset
from humaneva.utils import fetch_multiview, fetch
from common.utils import set_seed
from common.loss import mpjpe


class Args:
    dataset_path = 'data/data_3d_humaneva15.npz'
    dataset_2d_path = 'data/data_2d_humaneva15_gt.npz'
    subjects_train = 'Train/S1,Train/S2,Train/S3'.split(',')
    actions_train = 'Walk,Jog,Box'.split(',')
    subjects_val = 'Validate/S1,Validate/S2,Validate/S3'.split(',')
    actions_val = actions_train
    n_epochs = 150
    batch_size = 128
    wandb = False
    visualize_frame = True
    viz_dir = 'data/visuals/'
    model_dir = 'data/saved_models/'
    seed = 982356147
    # ckpt_path = 'data/saved_models/absurd-voice-21/last_checkpoint.pth'
    ckpt_path = None
    lr = 3e-4
    model_type = 'sequence' # or 'frame' or 'sequence'
    
    
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
    
    criterion = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scheduler = None
    
    if args.model_type == 'frame':
        from humaneva.train_frame import run
        poses_train_3d, poses_train_2d, cameras_train = fetch_multiview(args.subjects_train, keypoints, 
                                                                        he_dataset, args.actions_train)
        poses_val_3d, poses_val_2d, cameras_val = fetch_multiview(args.subjects_val, keypoints, 
                                                                  he_dataset, args.actions_val)
        train_dataset = MultiViewDataset(poses_train_2d, poses_train_3d, cameras_train, 
                                     keypoints_metadata, he_dataset.skeleton(), he_dataset.fps())
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        val_dataset = MultiViewDataset(poses_val_2d, poses_val_3d, cameras_val, 
                                     keypoints_metadata, he_dataset.skeleton(), he_dataset.fps())
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        if args.ckpt_path != None:
            model = torch.load(args.ckpt_path).to(device)
        else:
            model = FrameModel(n_joints=15, linear_size=1024, dropout=0.5, n_blocks=2).to(device)
        
    elif args.model_type == 'baseline':
        from humaneva.train_baseline import run
        criterion = mpjpe
        poses_train_3d, poses_train_2d, cameras_train = fetch(args.subjects_train, keypoints, 
                                                              he_dataset, args.actions_train)
        poses_val_3d, poses_val_2d, cameras_val = fetch(args.subjects_val, keypoints, 
                                                        he_dataset, args.actions_val)
        train_dataset = FrameDataset(poses_train_2d, poses_train_3d, cameras_train, 
                                        keypoints_metadata, he_dataset.skeleton(), he_dataset.fps())
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True)
        val_dataset = FrameDataset(poses_val_2d, poses_val_3d, cameras_val,
                                          keypoints_metadata, he_dataset.skeleton(), he_dataset.fps())
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False)
        if args.ckpt_path != None:
            model = torch.load(args.ckpt_path).to(device)
        else:
            model = FrameModel(n_joints=15, linear_size=1024, dropout=0.5).to(device)
        
    elif args.model_type == 'sequence':
        from humaneva.train_sequence import run
        poses_train_3d, poses_train_2d, cameras_train = fetch_multiview(args.subjects_train, keypoints, 
                                                                        he_dataset, args.actions_train)
        poses_val_3d, poses_val_2d, cameras_val = fetch_multiview(args.subjects_val, keypoints, 
                                                                  he_dataset, args.actions_val)
        train_dataset = MultiViewDataset(poses_train_2d, poses_train_3d, cameras_train, 
                                 keypoints_metadata, he_dataset.skeleton(), he_dataset.fps(), mode='sequence')
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)

        val_dataset = MultiViewDataset(poses_val_2d, poses_val_3d, cameras_val, 
                                     keypoints_metadata, he_dataset.skeleton(), he_dataset.fps(), mode='sequence')
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
        if args.ckpt_path != None:
            model = torch.load(args.ckpt_path).to(device)
        else:
            model = SequenceModel(n_joints=15, linear_size=512, n_encoder_heads=4, n_encoder_layers=2, dropout=0.1).to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    run(args.n_epochs, train_dataloader, val_dataloader, criterion, device, model, optimizer, scheduler=scheduler, 
        use_wandb=args.wandb, visualize_frame=args.visualize_frame, 
        dataset=val_dataset, model_output_dir=args.model_dir, viz_output_dir=args.viz_dir)
    
    torch.save(model, os.path.join(args.model_dir, 'last_checkpoint.pth'))
    return
    