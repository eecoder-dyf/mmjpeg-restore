import os
import argparse
import torch
import torch.nn.functional as F # 导入 F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import math
import cv2
from skimage.metrics import structural_similarity as ssim
from torchprofile import profile_macs

# 从项目文件中导入模型和数据加载器
from net import DB_ADMM_Net_RGB
from data import MultiModal_Dataset

def calculate_psnr(img1, img2, max_val=1.0):
    """计算两张图像之间的PSNR值 (tensor version)"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_val / math.sqrt(mse))

def calculate_ssim(img1, img2):
    """计算两张图像之间的SSIM值 (numpy version)"""
    # SSIM 需要 HWC 格式, 且数据范围为 [0, 1]
    img1_np = img1.squeeze().cpu().numpy().transpose(1, 2, 0)
    img2_np = img2.squeeze().cpu().numpy().transpose(1, 2, 0)
    return ssim(img1_np, img2_np, data_range=1.0, channel_axis=2, win_size=11, gaussian_weights=True, sigma=1.5)

def tensor2img(tensor):
    """将 tensor [C, H, W] 转换为 BGR uint8 图像 [H, W, C]"""
    img = tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img, 0, 1)
    img = (img * 255.0).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def pad_to_multiple(img, multiple):
    """Pad an image tensor so that its dimensions are a multiple of `multiple`."""
    h, w = img.shape[-2:]
    new_h = (h + multiple - 1) // multiple * multiple
    new_w = (w + multiple - 1) // multiple * multiple
    pad_h = new_h - h
    pad_w = new_w - w
    # The padding format is (pad_left, pad_right, pad_top, pad_bottom)
    img = F.pad(img, (0, pad_w, 0, pad_h), 'reflect')
    return img

def load_model_checkpoint(path, model, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model_state_dict = checkpoint['state_dict']
    else:
        model_state_dict = checkpoint
    model.load_state_dict(model_state_dict)

def main(args):
    # --- 路径设置 ---
    experiment_path = os.path.join('experiments', args.experiment_name)
    checkpoint_file = args.checkpoint or os.path.join(experiment_path, 'checkpoints', 'best_model.pth')
    results_path = os.path.join(experiment_path, f'results_qf{args.qf}')
    if args.save_images:
        os.makedirs(results_path, exist_ok=True)

    if not os.path.exists(checkpoint_file):
        print(f"Error: Checkpoint file not found at {checkpoint_file}")
        return

    # --- 设备设置 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 数据加载 (不裁剪) ---
    modalities = ['rgb', 'nir']
    test_dataset = MultiModal_Dataset(
        root_dir=args.data_root,
        modalities=modalities,
        patch_size=None,  # 关键：设置为 None 以加载完整图像
        is_train=False,
        jpeg_compress_modalities=args.jpeg_modalities,
        quality_min=args.qf,
        quality_max=args.qf
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    print(f"Found {len(test_dataset)} images for testing.")

    # --- 加载模型 ---
    model = DB_ADMM_Net_RGB(num_stages=args.num_stages, channels=3).to(device)
    load_model_checkpoint(checkpoint_file, model, device)
    model.eval()
    print(f"Model loaded from {checkpoint_file}")

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    input1 = torch.randn(1, 3, 256, 256).to(device)
    input2 = torch.randn(1, 3, 256, 256).to(device)
    inputs = (input1, input2)
    macs = profile_macs(model, args=inputs)
    print(f"Model Params: {params / 1e6:.2f} M")
    print(f"Model MACs: {macs / 1e9:.4f} G")

    # --- 测试循环 ---
    total_psnr_u, total_ssim_u = 0.0, 0.0
    total_psnr_v, total_ssim_v = 0.0, 0.0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f'Testing QF={args.qf}')
        for i, batch in enumerate(pbar):
            u_lq = batch['rgb_lq'].to(device)
            v_lq = batch['nir_lq'].to(device)
            u_gt = batch['rgb_gt'].to(device)
            v_gt = batch['nir_gt'].to(device)

            # 记录原始尺寸并进行 padding
            ori_h, ori_w = u_lq.shape[2:]
            u_lq = pad_to_multiple(u_lq, 4)
            v_lq = pad_to_multiple(v_lq, 4)
            # GT 也需要 padding 以便在计算 loss/metric 时尺寸匹配
            u_gt_padded = pad_to_multiple(u_gt, 4)
            v_gt_padded = pad_to_multiple(v_gt, 4)

            # 前向传播
            outputs = model(u_lq, v_lq)
            u_restored, v_restored, _, _ = outputs[-1]

            # 裁剪恢复的图像和 GT 至原始尺寸
            u_restored = u_restored[..., :ori_h, :ori_w]
            v_restored = v_restored[..., :ori_h, :ori_w]
            # 使用原始未 padding 的 GT 进行指标计算
            # u_gt 和 v_gt 已经是原始尺寸，无需裁剪

            # 计算指标
            psnr_u = calculate_psnr(u_restored, u_gt)
            ssim_u = calculate_ssim(u_restored, u_gt)
            psnr_v = calculate_psnr(v_restored, v_gt)
            ssim_v = calculate_ssim(v_restored, v_gt)

            total_psnr_u += psnr_u
            total_ssim_u += ssim_u
            total_psnr_v += psnr_v
            total_ssim_v += ssim_v

            # 保存图像
            if args.save_images:
                save_dirs = {
                    "rgb_restored": os.path.join(results_path, "rgb_restored"),
                    "nir_restored": os.path.join(results_path, "nir_restored"),
                    "rgb_lq": os.path.join(results_path, "rgb_lq"),
                    "nir_lq": os.path.join(results_path, "nir_lq"),
                    # "rgb_gt": os.path.join(results_path, "rgb_gt"),
                    # "nir_gt": os.path.join(results_path, "nir_gt"),
                }
                for path in save_dirs.values():
                    os.makedirs(path, exist_ok=True)
                base_filename = batch["filename"][0]
                # 保存的图像是裁剪后的
                cv2.imwrite(os.path.join(save_dirs["rgb_restored"], f"{base_filename}.png"), tensor2img(u_restored))
                cv2.imwrite(os.path.join(save_dirs["nir_restored"], f"{base_filename}.png"), tensor2img(v_restored))
                # 可选：保存输入和GT以供比较 (保存原始尺寸的)
                cv2.imwrite(os.path.join(save_dirs["rgb_lq"], f"{base_filename}.png"), tensor2img(batch["rgb_lq"]))
                cv2.imwrite(os.path.join(save_dirs["nir_lq"], f"{base_filename}.png"), tensor2img(batch["nir_lq"]))
                # cv2.imwrite(os.path.join(save_dirs["rgb_gt"], f"{base_filename}.png"), tensor2img(u_gt))
                # cv2.imwrite(os.path.join(save_dirs["nir_gt"], f"{base_filename}.png"), tensor2img(v_gt))

    # --- 打印平均结果 ---
    num_images = len(test_loader)
    avg_psnr_u = total_psnr_u / num_images
    avg_ssim_u = total_ssim_u / num_images
    avg_psnr_v = total_psnr_v / num_images
    avg_ssim_v = total_ssim_v / num_images

    print("\n--- Test Results ---")
    print(f"QF: {args.qf}")
    print(f"Modality 'rgb' (U):")
    print(f"  Avg PSNR: {avg_psnr_u:.2f} dB")
    print(f"  Avg SSIM: {avg_ssim_u:.4f}")
    print(f"Modality 'nir' (V):")
    print(f"  Avg PSNR: {avg_psnr_v:.2f} dB")
    print(f"  Avg SSIM: {avg_ssim_v:.4f}")
    if args.save_images:
        print(f"\nRestored images saved to: {results_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test DB-ADMM-Net for Multi-Modal JPEG Restoration')
    
    parser.add_argument('-exp', '--experiment_name', type=str, required=True, help='Name of the experiment to load the model from')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to a model checkpoint or full training checkpoint')
    parser.add_argument('--data_root', type=str, default=os.path.expanduser('~/database/RGB-NIR'), help='Root directory of the dataset')
    parser.add_argument('--num_stages', type=int, default=4, help='Number of stages in the network (must match the trained model)')
    
    # JPEG-related arguments
    parser.add_argument('--jpeg_modalities', nargs='+', default=['rgb', 'nir'], help='List of modalities to apply JPEG compression')
    parser.add_argument('--qf', type=int, default=40, help='Fixed JPEG quality factor for testing')
    
    # Other options
    parser.add_argument('--save_images', action='store_true', help='Set this flag to save restored images')
    
    args = parser.parse_args()
    main(args)