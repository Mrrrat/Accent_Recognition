import wandb
import torch
from torch import nn
import torchmetrics
from utils import SmoothCrossEntropyLoss
from tqdm import tqdm


def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is None or not p.requires_grad:
            continue
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def save(config, model, optimizer, scheduler=None, suffix='last'):
    if scheduler is not None:
        torch.save({
            'model_state_dict': model.state_dict(),
            #'opt_state_dict': optimizer.state_dict(),
            #'scheduler_state_dict': scheduler.state_dict()
            }, 'checkpoints/' + config['run_name'] + '_' + suffix + '.pt')
    else:
        torch.save({
            'model_state_dict': model.state_dict(),
            #'opt_state_dict': optimizer.state_dict(),
            }, 'checkpoints/' + config['run_name'] + '_' + suffix + '.pt')


def train(config, model, optimizer, train_loader, val_loader=None, scheduler=None):
    device = config['device']
    epochs = config['n_epochs'] 
    
    #criterion = nn.CrossEntropyLoss().to(device)
    criterion = SmoothCrossEntropyLoss(smoothing=config['smoothing']).to(device)
        
    best_loss = 1e10
    for epoch in tqdm(range(epochs)):
        train_epoch(model, optimizer, scheduler, train_loader, criterion, device, config['num_classes'], commit=val_loader is None)
        if val_loader:
            cur_loss, cur_acc = val_epoch(model, val_loader, criterion, device, config['num_classes'])
            best_loss = min(best_loss, cur_loss)
            if best_loss == cur_loss:
                save(config, model, optimizer, scheduler, 'best')
                #print(f'Epoch: {epoch} | Val Loss: {cur} | Saved')
        else:
            save(config, model, optimizer, scheduler, 'last')
      
        
def train_epoch(model, 
                optimizer, 
                scheduler, 
                loader, 
                criterion, 
                device, 
                num_classes, 
                commit=False, 
                use_mp=True, 
                accum_steps=1):
    model.train()
    total_loss, total_steps = 0, 0

    preds, targets = torch.tensor([]).to(device), torch.tensor([]).to(device)
    
    scaler = torch.cuda.amp.GradScaler() if use_mp else None
    
    for step, batch in enumerate(tqdm(loader)):
        data, target = batch[0].to(device), batch[1].to(device)
        
        if use_mp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred = model(data)
                loss = criterion(pred, target)
            total_loss += loss.item() * target.size(0)
            
            scaler.scale(loss).backward()
            grad_norm = get_grad_norm(model)
            if step % accum_steps == accum_steps - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            pred = model(data)
            loss = criterion(pred, target)
            total_loss += loss.item() * target.size(0)
            loss.backward()
            grad_norm = get_grad_norm(model)
            if step % accum_steps == accum_steps - 1:
                optimizer.step()
                optimizer.zero_grad()
        
        preds = torch.cat([preds, pred])
        targets = torch.cat([targets, target])
        total_steps += target.size(0)

    targets = targets.type(torch.cuda.IntTensor)
    acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)   
    
    if scheduler is not None:
        wandb.log({'Train_Accuracy': acc(preds, targets),
                   'Train_Loss': total_loss/total_steps,
                   'Learning_Rate': scheduler.get_last_lr()[0],
                   'Grad_Norm': grad_norm}, commit=commit)
        scheduler.step()
    else:
        wandb.log({'Train_Accuracy': acc(preds, targets),
                   'Train_Loss': total_loss/total_steps,
                   'Grad_Norm': grad_norm}, commit=commit)


@torch.no_grad()
def val_epoch(model, 
              loader, 
              criterion, 
              device, 
              num_classes, 
              use_mp=True, 
              accum_steps=1):
    model.eval()
    total_loss, total_steps = 0, 0

    preds, targets = torch.tensor([]).to(device), torch.tensor([]).to(device)
    
    scaler = torch.cuda.amp.GradScaler() if use_mp else None
    
    for batch in tqdm(loader):
        data, target = batch[0].to(device), batch[1].to(device)
        
        if use_mp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred = model(data)
                loss = criterion(pred, target)
            total_loss += loss.item() * target.size(0)
        else:
            pred = model(data)
            loss = criterion(pred, target)
            total_loss += loss.item() * target.size(0)
        
        preds = torch.cat([preds, pred])
        targets = torch.cat([targets, target])
        total_steps += target.size(0)

    targets = targets.type(torch.cuda.IntTensor)
    acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)   
    
    wandb.log({'Val_Accuracy': acc(preds, targets),
               'Val_Loss': total_loss/total_steps
              })
    
    return total_loss/total_steps, acc


@torch.no_grad()
def test_epoch(model,
              loader,
              device,
              num_classes,
              use_mp=True):
    model.eval()

    preds, targets = torch.tensor([]).to(device), torch.tensor([]).to(device)
    
    scaler = torch.cuda.amp.GradScaler() if use_mp else None
    
    for batch in tqdm(loader):
        data, target = batch[0].to(device), batch[1].to(device)
        
        if use_mp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred = model(data)
        else:
            pred = model(data)
        
        preds = torch.cat([preds, pred])
        targets = torch.cat([targets, target])

    targets = targets.type(torch.cuda.IntTensor)
    acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)   
    
    return acc(preds, targets)