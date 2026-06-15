import os
import cv2
import numpy as np
import argparse
import glob
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(img1, img2):
    """
    计算两张图像 (numpy 格式, 0-255) 之间的 PSNR
    """
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    """
    计算两张图像 (numpy 格式, 0-255, BGR) 之间的 SSIM
    """
    # ssim 函数要求多通道图像，且需要指定 channel_axis
    # data_range 是图像的动态范围
    return ssim(img1, img2, data_range=255.0, channel_axis=2, win_size=11, gaussian_weights=True, sigma=1.5)

def jpeg_compress(img_bgr, quality):
    """
    执行 JPEG 压缩并返回压缩后的图像 (BGR)
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img_bgr, encode_param)
    img_lq_bgr = cv2.imdecode(encimg, 1)
    return img_lq_bgr

def main():
    parser = argparse.ArgumentParser(description='Calculate average PSNR for JPEG compression')
    parser.add_argument('--input', type=str, required=True, help='输入图像文件夹路径')
    parser.add_argument('--qf', type=int, default=50, help='JPEG 质量因子 (1-100)')
    parser.add_argument('--ext', type=str, default='png', help='图像文件扩展名 (如 png, jpg)')
    
    args = parser.parse_args()

    # 获取所有图像路径
    img_paths = sorted(glob.glob(os.path.join(args.input, f'*.{args.ext}')))
    
    if not img_paths:
        # 尝试常用格式
        for ext in ['jpg', 'jpeg', 'bmp', 'tiff']:
            img_paths = sorted(glob.glob(os.path.join(args.input, f'*.{ext}')))
            if img_paths: break

    if not img_paths:
        print(f"在路径 {args.input} 下未找到图像文件。")
        return

    print(f"正在处理 {len(img_paths)} 张图像, QF = {args.qf}...")
    
    psnr_values = []
    ssim_values = []

    for path in tqdm(img_paths):
        # 1. 读取原始图像 (BGR)
        img_gt = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_gt is None:
            continue
            
        # 2. 模拟 JPEG 压缩
        img_lq = jpeg_compress(img_gt, args.qf)
        
        # 3. 计算 PSNR 和 SSIM
        psnr = calculate_psnr(img_gt, img_lq)
        ssim_val = calculate_ssim(img_gt, img_lq)
        psnr_values.append(psnr)
        ssim_values.append(ssim_val)

    # 4. 统计结果
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    print(f"\n--- 结果汇总 ---")
    print(f"处理总数: {len(psnr_values)}")
    print(f"固定 QF : {args.qf}")
    print(f"平均 PSNR: {avg_psnr:.4f} dB")
    print(f"平均 SSIM: {avg_ssim:.4f}")

if __name__ == '__main__':
    main()