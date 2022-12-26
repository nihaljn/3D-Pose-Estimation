import numpy as np
import os
import torch
import sys
from torch.utils.data import DataLoader
import wandb

from external.camera import *
from external.humaneva_dataset import HumanEvaDataset
from common.model import FrameModel
from common.dataset import MultiViewDataset
from common.utils import set_seed
from humaneva.utils import *
from common.loss import *
from external.visualization import visualize
from common.train_frame import train, validate


class Args:
    dataset_path = 'data/data_3d_humaneva15.npz'
    dataset_2d_path = 'data/data_2d_humaneva15_gt.npz'
    subjects_train = 'Train/S1,Train/S2,Train/S3'.split(',')
    actions_train = 'Walk,Jog,Box'.split(',')
    subjects_val = 'Validate/S1,Validate/S2,Validate/S3'.split(',')
    actions_val = actions_train
    n_epochs = 200
    batch_size = 64
    wandb = False
    visualize_frame = True
    viz_dir = 'data/visuals/'
    model_dir = 'data/saved_models/'
    checkpoint_fp = 'data/saved_models/fancy-frog-40/epoch_149.pth'
    seed = 982356147
    adapt_example = False
    
    
def adapt(n_epochs, model_output_dir, dataloader, device, model, optimizer, use_wandb=False):
    for epoch in range(n_epochs):
        # Training
        model.eval()
        criterion = None
        if model_output_dir != None:
            output_fp = os.path.join(model_output_dir, f'epoch_{epoch}.pth')
        step_cnt = [0]
        train_loss = train(n_epochs, epoch, step_cnt, dataloader, criterion, device, model, optimizer, output_fp)
        print(f'Epoch {epoch}/{n_epochs}\tStep {step_cnt[0]}\tTrain Loss {train_loss:.4}')
        if use_wandb:
            wandb.log({'train_loss': train_loss, 'epoch': epoch, 'step_cnt': step_cnt[0]})
        val_loss = validate(epoch, dataloader, criterion, device, model, visualize_frame=False)
        print(f'Epoch {epoch}/{n_epochs}\tStep {step_cnt[0]}\tValidation Loss {val_loss:.4}')
        if use_wandb:
            wandb.log({'val/loss': val_loss, 'epoch': epoch, 'step_cnt': step_cnt[0]})
    
    
def adapt_example(n_epochs, pose_2d, pose_3d, cameras, device, model, optimizer, scheduler, 
        use_wandb, visualize_frame, dataset, model_output_dir, viz_output_dir):
    
    cam0_2d, cam1_2d, cam2_2d = pose_2d[0].to(device), pose_2d[1].to(device), pose_2d[2].to(device)
    cameras[0], cameras[1], cameras[2] = cameras[0].to(device), cameras[1].to(device), cameras[2].to(device)
    n_batch = cameras[0].shape[0]
    # concatenating to pass as a single batch instead of 3 separate forward passes
    all_cam_2d = torch.cat((cam0_2d, cam1_2d, cam2_2d), dim=0)
    assert n_batch == 1
    
    for epoch in range(n_epochs):
        
        all_cam_3d_pred = model(all_cam_2d)
        # recovering per camera predictions
        cam_3d_preds = [all_cam_3d_pred[:1], all_cam_3d_pred[1:2], all_cam_3d_pred[2:]]
        cam_2d_preds = {(src, targ): None for src in range(3) for targ in range(3)} 
        
        # project 3d to 2d in each camera space
        for cam_idx_src in range(3):
            for cam_idx_targ in range(3):
                cam_2d_pred = camera_3d_to_camera_2d(cam_idx_src, cam_3d_preds[cam_idx_src], 
                                                     cam_idx_targ, cameras[cam_idx_src], cameras[cam_idx_targ], fix_offset=True)
                cam_2d_preds[(cam_idx_src, cam_idx_targ)] = cam_2d_pred
        
        # compute losses
        consistency_losses = []
        self_reconstruction_losses = []
        reconstruction_losses = []
        for cam_idx in range(3):
            consistency_losses.append(consistency_loss(cam_2d_preds[(0, cam_idx)], cam_2d_preds[(1, cam_idx)], cam_2d_preds[(2, cam_idx)]))
        
        for cam_idx_src in range(3):
            for cam_idx_targ in range(3):
                if cam_idx_src == cam_idx_targ:
                    self_reconstruction_losses.append(mpjpe(cam_2d_preds[(cam_idx_src, cam_idx_targ)], pose_2d[cam_idx_targ]))
                else:
                    reconstruction_losses.append(mpjpe(cam_2d_preds[(cam_idx_src, cam_idx_targ)], pose_2d[cam_idx_targ]))
                    
        consistency_loss_val = sum(consistency_losses) / len(consistency_losses)
        reconstruction_loss_val = sum(reconstruction_losses) / len(reconstruction_losses)
        self_reconstruction_loss_val = sum(self_reconstruction_losses) / len(self_reconstruction_losses)
        
        train_loss = consistency_loss_val + reconstruction_loss_val + self_reconstruction_loss_val
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        val_losses = []
        for cam_idx in range(3):
            val_losses.append(mpjpe(cam_3d_preds[cam_idx], pose_3d[cam_idx]))
        val_loss = sum(val_losses) / len(val_losses)
        
        print(f'Epoch {epoch}/{n_epochs}\tTrain Loss {train_loss.cpu().item():.4}')
        if use_wandb:
            wandb.log({'train/loss': train_loss.cpu().item(), 'epoch': epoch})
        print(f'Epoch {epoch}/{n_epochs}\tValidation Loss {val_loss.cpu().item():.4}')
        if use_wandb:
            wandb.log({'val/loss': val_loss.cpu().item(), 'epoch': epoch})
        
        output_fp = os.path.join(model_output_dir, f'epoch_{epoch}.pth')
        torch.save(model, output_fp)
        
        if visualize_frame:
            assert dataset != None, 'Need dataset object to visualize'
            assert viz_output_dir != None, 'Need path to visualization output file'
            output_fp = os.path.join(viz_output_dir, f'epoch_{epoch}.gif')
            idx = 0
            data_2d = pose_2d[0][idx].unsqueeze(0).numpy()
            pred_3d = cam_3d_preds[0][idx].detach().unsqueeze(0).cpu().numpy()
            targ_3d = pose_3d[0][idx].unsqueeze(0).numpy()
            cam = convert_cam_to_viz_dict(cameras[0][idx], 0)
            visualize(data_2d.copy(), targ_3d.copy(), pred_3d.copy(), 
                      dataset.keypoints_metadata, cam, dataset.skeleton, 
                      dataset.fps, output_fp=output_fp)
    
    
def main():
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
                
    poses_train_3d, poses_train_2d, cameras_train = fetch_multiview(args.subjects_train, keypoints, he_dataset, args.actions_train)
    poses_val_3d, poses_val_2d, cameras_val = fetch_multiview(args.subjects_val, keypoints, he_dataset, args.actions_val)
    
    criterion = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(args.checkpoint_fp).to(device)
    model.eval() # test time adaptation in eval mode
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.7)
    scheduler = None
    
    if args.adapt_example:
        val_dataset = MultiViewDataset(poses_val_2d, poses_val_3d, cameras_val, 
                                     keypoints_metadata, he_dataset.skeleton(), he_dataset.fps())
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=2)
        # pick an example
        for pose_2d, pose_3d, cameras in val_dataloader:
            break
        adapt_example(args.n_epochs, pose_2d, pose_3d, cameras, device, model, optimizer, scheduler=scheduler, 
            use_wandb=args.wandb, visualize_frame=args.visualize_frame, 
            dataset=val_dataset, model_output_dir=args.model_dir, viz_output_dir=args.viz_dir)
    
    else:
        val_dataset = MultiViewDataset(poses_val_2d, poses_val_3d, cameras_val, 
                                     keypoints_metadata, he_dataset.skeleton(), he_dataset.fps())
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        adapt(args.n_epochs, args.model_dir, val_dataloader, device, model, optimizer, use_wandb=args.wandb)
    
    torch.save(model, os.path.join(args.model_dir, 'last_checkpoint.pth'))
    return
    
    
if __name__ == '__main__':
    main()
    