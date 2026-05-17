import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import math

# 从项目文件中导入模型和数据加载器
from net import DB_ADMM_Net_RGB
from data import MultiModal_Dataset


def get_base_model(model):
    """Return the underlying model when wrapped by DataParallel."""
    return model.module if isinstance(model, nn.DataParallel) else model


def adapt_state_dict_for_model(model, state_dict):
    """Adapt checkpoint keys between DataParallel and non-DataParallel formats."""
    base_model = get_base_model(model)
    model_keys = list(base_model.state_dict().keys())
    ckpt_keys = list(state_dict.keys())

    if not model_keys or not ckpt_keys:
        return state_dict

    model_has_module_prefix = model_keys[0].startswith('module.')
    ckpt_has_module_prefix = ckpt_keys[0].startswith('module.')

    if model_has_module_prefix == ckpt_has_module_prefix:
        return state_dict

    if ckpt_has_module_prefix and not model_has_module_prefix:
        return {k[len('module.'):]: v for k, v in state_dict.items()}

    if not ckpt_has_module_prefix and model_has_module_prefix:
        return {f'module.{k}': v for k, v in state_dict.items()}

    return state_dict

def calculate_psnr(img1, img2, max_val=1.0):
    """计算两张图像之间的PSNR值"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_val / math.sqrt(mse))


def train_epoch(model, loader, optimizer, loss_fn, device, stage_weights, alpha):
    """
    执行一个训练周期的函数
    """
    model.train()
    epoch_loss = 0.0
    epoch_prior_loss = 0.0
    
    pbar = tqdm(loader, desc='Training', leave=False, dynamic_ncols=True)
    for batch in pbar:
        # 根据新的数据格式解包
        # 假设 U=rgb, V=nir
        u_lq = batch['rgb_lq'].to(device)
        v_lq = batch['nir_lq'].to(device)
        u_gt = batch['rgb_gt'].to(device)
        v_gt = batch['nir_gt'].to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播，输入为低质量图像
        outputs = model(u_lq, v_lq)

        base_model = get_base_model(model)
        p_values = base_model.get_p_values()
        
        # 计算损失，与高质量图像比较
        total_loss = 0
        prior_loss = 0
        num_stages = len(outputs)
        for k in range(num_stages):
            u_k, v_k, c_map = outputs[k]
            # L1 Loss
            loss_u = loss_fn(u_k, u_gt)
            loss_v = loss_fn(v_k, v_gt)
            total_loss += stage_weights[k] * (loss_u + loss_v)

            # Charbonnier prior loss
            p_k = p_values[k]
            eps = 1e-4
            b, c, h, w = c_map.shape
            prior_k = torch.sum((c_map ** 2 + eps ** 2) ** (p_k / 2.0)) / (b * h * w)
            prior_loss += prior_k

        total_loss = total_loss + alpha * prior_loss
            
        # 反向传播和优化
        total_loss.backward()
        optimizer.step()
        
        epoch_loss += total_loss.item()
        epoch_prior_loss += prior_loss.item()
        pbar.set_postfix(loss=total_loss.item(), prior=prior_loss.item())
        
    return epoch_loss / len(loader), epoch_prior_loss / len(loader)

def test_epoch(model, loader, loss_fn, device, stage_weights, alpha):
    """
    执行一个验证/测试周期的函数
    """
    model.eval()
    epoch_loss = 0.0
    epoch_prior_loss = 0.0
    total_psnr_u = 0.0
    total_psnr_v = 0.0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation', leave=False, dynamic_ncols=True)
        for batch in pbar:
            # 根据新的数据格式解包
            u_lq = batch['rgb_lq'].to(device)
            v_lq = batch['nir_lq'].to(device)
            u_gt = batch['rgb_gt'].to(device)
            v_gt = batch['nir_gt'].to(device)
            
            # 前向传播
            outputs = model(u_lq, v_lq)
            base_model = get_base_model(model)
            p_values = base_model.get_p_values()
            
            # 只评估最后一个阶段的输出
            u_final, v_final, _ = outputs[-1]
            
            # 计算损失
            total_loss = 0
            prior_loss = 0
            num_stages = len(outputs)
            for k in range(num_stages):
                u_k, v_k, c_map = outputs[k]
                loss_u = loss_fn(u_k, u_gt)
                loss_v = loss_fn(v_k, v_gt)
                total_loss += stage_weights[k] * (loss_u + loss_v)

                p_k = p_values[k]
                eps = 1e-4
                b, c, h, w = c_map.shape
                prior_k = torch.sum((c_map ** 2 + eps ** 2) ** (p_k / 2.0)) / (b * h * w)
                prior_loss += prior_k

            total_loss = total_loss + alpha * prior_loss
            
            epoch_loss += total_loss.item()
            epoch_prior_loss += prior_loss.item()
            
            # 计算最终输出的 PSNR
            psnr_u = calculate_psnr(u_final, u_gt)
            psnr_v = calculate_psnr(v_final, v_gt)
            total_psnr_u += psnr_u
            total_psnr_v += psnr_v
            
            pbar.set_postfix(psnr_u=f"{psnr_u:.2f}", psnr_v=f"{psnr_v:.2f}", prior=f"{prior_loss.item():.3f}")

    avg_loss = epoch_loss / len(loader)
    avg_psnr_u = total_psnr_u / len(loader)
    avg_psnr_v = total_psnr_v / len(loader)
    
    avg_prior = epoch_prior_loss / len(loader)
    return avg_loss, avg_psnr_u, avg_psnr_v, avg_prior

def load_checkpoint(path, model, device, optimizer=None, scheduler=None, load_training_state=False):
    checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
        resumed_epoch = checkpoint.get('epoch')
        best_psnr = checkpoint.get('best_psnr')

        if load_training_state:
            optimizer_state_dict = checkpoint.get('optimizer_state_dict')
            scheduler_state_dict = checkpoint.get('scheduler_state_dict')
            if optimizer_state_dict is not None and scheduler_state_dict is not None and optimizer is not None and scheduler is not None:
                optimizer.load_state_dict(optimizer_state_dict)
                scheduler.load_state_dict(scheduler_state_dict)
            else:
                print('Warning: optimizer/scheduler state not found in checkpoint.')
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model_state_dict = checkpoint['state_dict']
        resumed_epoch = checkpoint.get('epoch')
        best_psnr = checkpoint.get('best_psnr')
    else:
        model_state_dict = checkpoint
        resumed_epoch = None
        best_psnr = None

    base_model = get_base_model(model)
    model_state_dict = adapt_state_dict_for_model(base_model, model_state_dict)
    missing_keys, unexpected_keys = base_model.load_state_dict(model_state_dict, strict=False)
    if missing_keys or unexpected_keys:
        print(f"Warning: load_state_dict missing={missing_keys}, unexpected={unexpected_keys}")
    return resumed_epoch, best_psnr

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_psnr):
    base_model = get_base_model(model)
    torch.save({
        'epoch': epoch,
        'best_psnr': best_psnr,
        'model_state_dict': base_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, path)

def main(args):
    # --- 实验路径设置 ---
    experiment_path = os.path.join('experiments', args.experiment_name)
    checkpoint_path = os.path.join(experiment_path, 'checkpoints')
    tensorboard_path = os.path.join(experiment_path, 'logs')
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(tensorboard_path, exist_ok=True)

    # --- TensorBoard ---
    writer = SummaryWriter(log_dir=tensorboard_path)

    # --- 设备设置 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 数据加载 ---
    modalities = ['rgb', 'nir']
    train_dataset = MultiModal_Dataset(
        root_dir=args.data_root,
        modalities=modalities,
        patch_size=args.patch_size,
        is_train=True,
        jpeg_compress_modalities=args.jpeg_modalities,
        quality_min=args.qf,
        quality_max=args.qf
    )
    val_dataset = MultiModal_Dataset(
        root_dir=args.data_root,
        modalities=modalities,
        patch_size=args.val_patch_size,
        is_train=False,
        jpeg_compress_modalities=args.jpeg_modalities,
        quality_min=args.qf,
        quality_max=args.qf
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # --- 模型、损失函数、优化器 ---
    p_value = args.p
    model = DB_ADMM_Net_RGB(num_stages=args.num_stages, channels=3, p_value=p_value).to(device)
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, mode='min', patience=6, cooldown=6, min_lr=1e-6)

    stage_weights = [1.0] * args.num_stages
    start_epoch = args.start_epoch
    saved_best_psnr = None

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint: {args.resume}")
            resumed_epoch, saved_best_psnr = load_checkpoint(
                args.resume,
                model,
                device,
                optimizer=optimizer,
                scheduler=scheduler,
                load_training_state=args.resume_state,
            )
            if resumed_epoch is not None and args.resume_state:
                start_epoch = resumed_epoch + 1
                print(f"Resuming training from epoch {start_epoch}")
            if args.p is not None:
                base_model = get_base_model(model)
                base_model.p_fixed.fill_(float(args.p))
        else:
            print(f"Warning: Checkpoint not found at {args.resume}. Training from scratch.")

    # --- 初始验证 (作为基准) ---
    print()
    print("--- Initial Validation (Before Training) ---")
    initial_val_loss, initial_psnr_u, initial_psnr_v, initial_prior_loss = test_epoch(
        model, val_loader, loss_fn, device, stage_weights, args.alpha
    )
    initial_avg_psnr = (initial_psnr_u + initial_psnr_v) / 2
    best_psnr = saved_best_psnr if saved_best_psnr is not None else initial_avg_psnr

    base_model = get_base_model(model)
    p_values = base_model.get_p_values()
    p1 = p_values[0].item()
    pK = p_values[-1].item()

    print(f"Initial Val Loss: {initial_val_loss:.4f}")
    print(f"Initial Prior Loss: {initial_prior_loss:.4f}")
    print(f"Initial Val PSNR (U): {initial_psnr_u:.2f} dB")
    print(f"Initial Val PSNR (V): {initial_psnr_v:.2f} dB")
    print(f"Current Best PSNR: {best_psnr:.2f} dB")
    print(f"Initial p values: p1={p1:.3f}, pK={pK:.3f}")

    # 记录初始指标
    initial_step = max(0, start_epoch - 1)
    writer.add_scalar('Loss/val', initial_val_loss, initial_step)
    writer.add_scalar('Loss/prior_val', initial_prior_loss, initial_step)
    writer.add_scalar('PSNR/val_U', initial_psnr_u, initial_step)
    writer.add_scalar('PSNR/val_V', initial_psnr_v, initial_step)
    writer.add_scalar('PSNR/val_avg', initial_avg_psnr, initial_step)

    # --- 训练循环 ---
    for epoch in range(start_epoch, args.epochs + 1):
        print()
        print(f"--- Epoch {epoch}/{args.epochs} ---")

        train_loss, train_prior_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, device, stage_weights, args.alpha
        )
        val_loss, val_psnr_u, val_psnr_v, val_prior_loss = test_epoch(
            model, val_loader, loss_fn, device, stage_weights, args.alpha
        )
        avg_val_psnr = (val_psnr_u + val_psnr_v) / 2

        base_model = get_base_model(model)
        p_values = base_model.get_p_values()
        p1 = p_values[0].item()
        pK = p_values[-1].item()

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        print(f"Epoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Train Prior Loss: {train_prior_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Prior Loss:   {val_prior_loss:.4f}")
        print(f"  Val PSNR (U): {val_psnr_u:.2f} dB")
        print(f"  Val PSNR (V): {val_psnr_v:.2f} dB")
        print(f"  p values: p1={p1:.3f}, pK={pK:.3f}")

        # 记录到 TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/prior_train', train_prior_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Loss/prior_val', val_prior_loss, epoch)
        writer.add_scalar('PSNR/val_U', val_psnr_u, epoch)
        writer.add_scalar('PSNR/val_V', val_psnr_v, epoch)
        writer.add_scalar('PSNR/val_avg', avg_val_psnr, epoch)
        writer.add_scalar('learning_rate', current_lr, epoch)
        writer.add_scalar('p_values/p1', p1, epoch)
        writer.add_scalar('p_values/pK', pK, epoch)

        if avg_val_psnr > best_psnr:
            best_psnr = avg_val_psnr
            save_path = os.path.join(checkpoint_path, 'best_model.pth')
            torch.save(get_base_model(model).state_dict(), save_path)
            print(f"  New best model saved to {save_path} (PSNR: {best_psnr:.2f} dB)")

        save_checkpoint(
            os.path.join(checkpoint_path, 'latest_checkpoint.pth'),
            model,
            optimizer,
            scheduler,
            epoch,
            best_psnr,
        )

    writer.close()
    print("Training finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DB-ADMM-Net for Multi-Modal JPEG Restoration')

    parser.add_argument('-exp', '--experiment_name', type=str, required=True, help='Name for the experiment')
    parser.add_argument('--data_root', type=str, default=os.path.expanduser('~/database/RGB-NIR'), help='Root directory of the dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--start_epoch', type=int, default=1, help='Epoch to start training from (for resuming)')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--patch_size', type=int, default=128, help='Image patch size for training')
    parser.add_argument('--val_patch_size', type=int, default=None, help='Patch size for validation (center crop). If None, use full image.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_stages', type=int, default=4, help='Number of stages in the network')
    parser.add_argument('--p', type=float, default=None, help='Fixed p value for prior (if omitted, keep checkpoint value when resuming)')
    parser.add_argument('--alpha', type=float, default=0.01, help='Weight for prior loss')
    parser.add_argument('--resume', type=str, default=None, help='Path to a model checkpoint or full training checkpoint')
    parser.add_argument('--resume_state', action='store_true', help='Restore optimizer and scheduler states if available')

    # JPEG-related arguments
    parser.add_argument('--jpeg_modalities', nargs='+', default=['rgb', 'nir'], help='List of modalities to apply JPEG compression (e.g., rgb nir)')
    parser.add_argument('--qf', type=int, default=40, help='Fixed JPEG quality factor for compression')
    
    args = parser.parse_args()
    main(args)