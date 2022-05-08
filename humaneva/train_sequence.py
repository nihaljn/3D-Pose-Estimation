import os
import torch
import wandb

from external.visualization import visualize
from external.camera import camera_3d_to_camera_2d
from humaneva.utils import convert_cam_to_viz_dict
from common.loss import *


def validate(epoch, dataloader, criterion, device, model, visualize_frame=False, dataset=None, output_fp=None):
    total_loss = 0
    batch_cnt = 0
    for pose_2d, pose_3d, cameras in dataloader:
        
        assert pose_2d.shape[0] == 1, 'Batch size should be 1'
        pose_2d, pose_3d, cameras = pose_2d.to(device), pose_3d.to(device), cameras.squeeze().to(device)
        n_frames = cameras.shape[1]
        
        pose_3d_pred = [None, None, None]
        # mask = torch.full((n_frames, n_frames), -torch.inf).to(device)
        # attend_size = 5
        # for i in range(n_frames):
        #     mask[i, max(i - attend_size, 0):min(i + attend_size + 1, n_frames)] = 0
        mask = None
        loss = 0
        for cam_idx in range(3):
            with torch.no_grad():
                pose_3d_pred[cam_idx] = model(pose_2d[:, cam_idx], src_mask=mask) # (n_frames x n_joints x 3)
            loss += mpjpe(pose_3d_pred[cam_idx], pose_3d[0, cam_idx]) 
        
        total_loss += (loss/3).cpu().item()
        batch_cnt += 1
    
    if visualize_frame:
        assert dataset != None, 'Need dataset object to visualize'
        assert output_fp != None, 'Need path to visualization output file'
        idx = min(12, pose_2d[0].shape[0] - 1)
        cam_idx = 1
        # print(pose_2d.shape, pose_3d_pred[0].shape, cameras.shape)
        data_2d = pose_2d[0, cam_idx][idx].unsqueeze(0).numpy()
        pred_3d = pose_3d_pred[cam_idx][idx].unsqueeze(0).cpu().numpy()
        targ_3d = pose_3d[0, cam_idx][idx].unsqueeze(0).numpy()
        cam = convert_cam_to_viz_dict(cameras[cam_idx][idx], cam_idx)
        visualize(data_2d.copy(), targ_3d.copy(), pred_3d.copy(), 
                  dataset.keypoints_metadata, cam, dataset.skeleton, 
                  dataset.fps, output_fp=output_fp)
    
    return total_loss / batch_cnt


def train(n_epochs, epoch, step_cnt, dataloader, criterion, device, model, optimizer, output_fp=None):
    total_loss = 0
    batch_cnt = 0
    for pose_2d, pose_3d, cameras in dataloader:
        
        assert pose_2d.shape[0] == 1, 'Batch size should be 1'
        pose_2d, pose_3d, cameras = pose_2d.to(device), pose_3d.to(device), cameras.squeeze().to(device)
        n_frames = cameras.shape[1]
        
        pose_3d_pred = [None, None, None]
        # mask = torch.full((n_frames, n_frames), -torch.inf).to(device)
        # attend_size = 5
        # for i in range(n_frames):
        #     mask[i, max(i - attend_size, 0):min(i + attend_size + 1, n_frames)] = 0
        mask = None
        for cam_idx in range(3):
            pose_3d_pred[cam_idx] = model(pose_2d[:, cam_idx], src_mask=mask) # (n_frames x n_joints x 3)
        cam_2d_preds = {(src, targ): [] for src in range(3) for targ in range(3)}
        
        # project 3d to 2d in each camera space
        for idx in range(n_frames):
            cams = [cameras[0][idx:idx+1], cameras[1][idx:idx+1], cameras[2][idx:idx+1]]
            for cam_idx_src in range(3):
                for cam_idx_targ in range(3):
                    cam_2d_pred = camera_3d_to_camera_2d(cam_idx_src, pose_3d_pred[cam_idx_src][idx:idx+1], 
                                                         cam_idx_targ, cams[cam_idx_src], cams[cam_idx_targ], fix_offset=True)
                    cam_2d_preds[(cam_idx_src, cam_idx_targ)].append(cam_2d_pred)
        cam_2d_preds = {k: torch.cat(v, dim=0) for k, v in cam_2d_preds.items()}
        
        # compute losses
        consistency_losses = []
        self_reconstruction_losses = []
        reconstruction_losses = []
        for cam_idx in range(3):
            consistency_losses.append(consistency_loss(cam_2d_preds[(0, cam_idx)], cam_2d_preds[(1, cam_idx)], cam_2d_preds[(2, cam_idx)]))
        
        for cam_idx_src in range(3):
            for cam_idx_targ in range(3):
                if cam_idx_src == cam_idx_targ:
                    self_reconstruction_losses.append(mpjpe(cam_2d_preds[(cam_idx_src, cam_idx_targ)], pose_2d[0, cam_idx_targ]))
                else:
                    reconstruction_losses.append(mpjpe(cam_2d_preds[(cam_idx_src, cam_idx_targ)], pose_2d[0, cam_idx_targ]))
                    
        consistency_loss_val = sum(consistency_losses) / len(consistency_losses)
        reconstruction_loss_val = sum(reconstruction_losses) / len(reconstruction_losses)
        self_reconstruction_loss_val = sum(self_reconstruction_losses) / len(self_reconstruction_losses)
        
        loss = consistency_loss_val + reconstruction_loss_val + self_reconstruction_loss_val
        if batch_cnt == len(dataloader) - 1:
            print(f'Consistency loss: {consistency_loss_val.cpu().item()}',
                  f'\tReconstruction Loss: {reconstruction_loss_val.cpu().item()}',
                  f'\tSelf-reconstruction loss: {self_reconstruction_loss_val.cpu().item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_cnt[0] += 1
        total_loss += loss.cpu().item()
        batch_cnt += 1
        
        torch.save(model, output_fp)
        
    return total_loss / batch_cnt
    
    
def run(n_epochs, train_loader, val_loader, criterion, device, model, optimizer, scheduler=None, use_wandb=False, 
        visualize_frame=False, dataset=None, model_output_dir=None, viz_output_dir=None):
    '''Train + Val'''
    step_cnt = [0] # as list to pass by reference
    
    for epoch in range(n_epochs):
        
        # Training
        if epoch <= n_epochs // 2:
            model.train()
        else:
            model.eval()
        # model.train()
            
        if model_output_dir != None:
            output_fp = os.path.join(model_output_dir, f'epoch_{epoch}.pth')
        train_loss = train(n_epochs, epoch, step_cnt, train_loader, criterion, device, model, optimizer, output_fp)
        print(f'Epoch {epoch}/{n_epochs}\tStep {step_cnt[0]}\tTrain Loss {train_loss:.4}')
        if use_wandb:
            wandb.log({'train/loss': train_loss, 'epoch': epoch, 'step_cnt': step_cnt[0]})
        
        # Validation
        model.eval()
        if viz_output_dir != None:
            output_fp = os.path.join(viz_output_dir, f'epoch_{epoch}.gif')
        val_loss = validate(epoch, val_loader, criterion, device, model, visualize_frame, dataset, output_fp)
        print(f'Epoch {epoch}/{n_epochs}\tStep {step_cnt[0]}\tValidation Loss {val_loss:.4}')
        if use_wandb:
            wandb.log({'val/loss': val_loss, 'epoch': epoch, 'step_cnt': step_cnt[0]})
        
        if scheduler != None:
            scheduler.step()
            
