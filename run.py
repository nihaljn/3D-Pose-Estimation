import os
import torch
import wandb

from external.visualization import visualize
from utils import convert_cam_to_viz_dict
from loss import mpjpe


def validate(epoch, dataloader, criterion, device, model, visualize_frame=False, dataset=None, output_fp=None):
    total_loss = 0
    batch_cnt = 0
    for pose_2d, pose_3d, cameras in dataloader:
        cam0_2d, cam1_2d, cam2_2d = pose_2d[0].to(device), pose_2d[1].to(device), pose_2d[2].to(device)
        with torch.no_grad():
            cam0_3d_pred = model(cam0_2d)
            cam1_3d_pred = model(cam1_2d)
            cam2_3d_pred = model(cam2_2d)
        cam0_3d_targ, cam1_3d_targ, cam2_3d_targ = pose_3d[0].to(device), pose_3d[1].to(device), pose_3d[2].to(device)
        loss = mpjpe(cam0_3d_pred, cam0_3d_targ) + mpjpe(cam1_3d_pred, cam1_3d_targ) + mpjpe(cam2_3d_pred, cam2_3d_targ)
        total_loss += (loss/3).cpu().item()
        batch_cnt += 1
    
    if visualize_frame:
        assert dataset != None, 'Need dataset object to visualize'
        assert output_fp != None, 'Need path to visualization output file'
        idx = 12
        data_2d = pose_2d[0][idx].unsqueeze(0).numpy()
        pred_3d = cam0_3d_pred[idx].unsqueeze(0).cpu().numpy()
        targ_3d = pose_3d[0][idx].unsqueeze(0).numpy()
        cam = convert_cam_to_viz_dict(cameras[0][idx], 0)
        visualize(data_2d.copy(), targ_3d.copy(), pred_3d.copy(), 
                  dataset.keypoints_metadata, cam, dataset.skeleton, 
                  dataset.fps, output_fp=output_fp)
    
    return total_loss / batch_cnt


def train(n_epochs, epoch, step_cnt, dataloader, criterion, device, model, optimizer):
    total_loss = 0
    batch_cnt = 0
    for pose_2d, pose_3d, cameras in dataloader:
        cam0_2d, cam1_2d, cam2_2d = pose_2d[0].to(device), pose_2d[1].to(device), pose_2d[2].to(device)
        cameras[0], cameras[1], cameras[2] = cameras[0].to(device), cameras[1].to(device), cameras[2].to(device)
        n_batch = cam0_2d.shape[0]
        all_cam_2d = torch.cat((cam0_2d, cam1_2d, cam2_2d), dim=0)
        all_cam_3d_pred = model(all_cam_2d)
        cam0_3d_pred = all_cam_3d_pred[:n_batch]
        cam1_3d_pred = all_cam_3d_pred[n_batch:2*n_batch]
        cam2_3d_pred = all_cam_3d_pred[2*n_batch:]
        
        losses = []
        for idx in range(cam0_2d.shape[0]):
            cam0, cam1, cam2 = cameras[0][idx:idx+1], cameras[1][idx:idx+1], cameras[2][idx:idx+1]
            cam_idx = torch.randint(0, 3, (1, )).item()
            if cam_idx == 0:
                cur_loss = criterion(cam0_3d_pred[idx:idx+1], cam1_2d[idx:idx+1], cam2_2d[idx:idx+1], cam0, cam1, cam2)
                losses.append(cur_loss)
            elif cam_idx == 1:
                cur_loss = criterion(cam1_3d_pred[idx:idx+1], cam0_2d[idx:idx+1], cam2_2d[idx:idx+1], cam1, cam0, cam2)
                losses.append(cur_loss)
            else:
                cur_loss = criterion(cam2_3d_pred[idx:idx+1], cam0_2d[idx:idx+1], cam1_2d[idx:idx+1], cam2, cam0, cam1)
                losses.append(cur_loss)
        
        loss = sum(losses) / len(losses)    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_cnt[0] += 1
        total_loss += loss.cpu().item()
        batch_cnt += 1
    return total_loss / batch_cnt
    
    
def run(n_epochs, train_loader, val_loader, criterion, device, model, optimizer, use_wandb=False, 
        visualize_frame=False, dataset=None, output_dir=None):
    '''Train + Val'''
    step_cnt = [0] # as list to pass by reference
    
    for epoch in range(n_epochs):
        
        # Training
        model.train()
        train_loss = train(n_epochs, epoch, step_cnt, train_loader, criterion, device, model, optimizer)
        print(f'Epoch {epoch}/{n_epochs}\tStep {step_cnt[0]}\tTrain Loss {train_loss:.4}')
        if use_wandb:
            wandb.log({'train/loss': train_loss, 'epoch': epoch, 'step_cnt': step_cnt[0]})
            
        # # Validation
        model.eval()
        if output_dir != None:
            output_fp = os.path.join(output_dir, f'epoch_{epoch}.gif')
        val_loss = validate(epoch, val_loader, criterion, device, model, visualize_frame, dataset, output_fp)
        print(f'Epoch {epoch}/{n_epochs}\tStep {step_cnt[0]}\tValidation Loss {val_loss:.4}')
        if use_wandb:
            wandb.log({'val/loss': train_loss, 'epoch': epoch, 'step_cnt': step_cnt[0]})