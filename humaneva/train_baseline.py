import os
import torch
from external.visualization import visualize
import wandb

from common.loss import mpjpe

def validate(epoch, dataloader, criterion, device, model, visualize_frame=False, dataset=None, output_fp=None):
    total_loss = 0
    batch_cnt = 0
    for batch in dataloader:
        model_inp = batch[0].to(device)
        target = batch[1].to(device)
        with torch.no_grad():
            preds = model(model_inp)
        loss = mpjpe(preds, target)
        total_loss += loss.cpu().item()
        batch_cnt += 1
    
    if visualize_frame:
        assert dataset != None, 'Need dataset object to visualize'
        assert output_fp != None, 'Need path to visualization output file'
        idx = 12
        data_2d = batch[0][idx].unsqueeze(0).numpy()
        pred_3d = preds[idx].unsqueeze(0).cpu().numpy()
        targ_3d = batch[1][idx].unsqueeze(0).numpy()
        cam = {key: val[idx] for key, val in batch[2].items()}
        cam['orientation'] = cam['orientation'].numpy()
        cam['translation'] = cam['translation'].numpy()
        cam['res_w'] = cam['res_w'].item()
        cam['res_h'] = cam['res_h'].item()
        cam['azimuth'] = cam['azimuth'].item()
        visualize(data_2d.copy(), targ_3d.copy(), pred_3d.copy(), 
                  dataset.keypoints_metadata, cam, dataset.skeleton, 
                  dataset.fps, output_fp=output_fp)
    
    return total_loss / batch_cnt


def train(n_epochs, epoch, step_cnt, dataloader, criterion, device, model, optimizer, weighted=False, output_fp=None):
    total_loss = 0
    batch_cnt = 0
    for batch in dataloader:
        model_inp = batch[0].to(device)
        target = batch[1].to(device)
        preds = model(model_inp)
        if not weighted:
            loss = criterion(preds, target)
        else:
            w = 1 / torch.abs(target[:, :, 2]).clone() # inversely proportional to the depth
            loss = criterion(preds, target, w)
        # print(model_inp.shape, target.shape, preds.shape)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_cnt[0] += 1
        total_loss += loss.cpu().item()
        batch_cnt += 1
    torch.save(model, output_fp)
    return total_loss / batch_cnt
    
    
def run(n_epochs, train_loader, val_loader, criterion, device, model, optimizer, use_wandb=False, scheduler=None,
        visualize_frame=False, dataset=None, model_output_dir=None, viz_output_dir=None, weighted=False):
    '''Train + Val'''
    step_cnt = [0] # as list to pass by reference
    
    for epoch in range(n_epochs):
        
        # Training
        if epoch <= n_epochs // 2:
            model.train()
        else:
            model.eval()
            
        if model_output_dir != None:
            output_fp = os.path.join(model_output_dir, f'epoch_{epoch}.pth')
        train_loss = train(n_epochs, epoch, step_cnt, train_loader, criterion, device, model, optimizer, weighted, output_fp)
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