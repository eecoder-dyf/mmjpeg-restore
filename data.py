import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
import random
import glob
from collections import defaultdict

# ==========================================
#  工具函数 (与 data.py 相同)
# ==========================================

def augment(img_list, hflip=True, rot=True):
    """数据增强：随机水平翻转和旋转"""
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]

def np2tensor(img_list):
    """numpy (H, W, C) -> tensor (C, H, W)"""
    def _to_tensor(img):
        # 如果是单通道灰度图，增加一个通道维度
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        return torch.from_numpy(np.ascontiguousarray(img.transpose((2, 0, 1)))).float()
    
    return [_to_tensor(img) for img in img_list]

def jpeg_compress(img, quality):
    """
    模拟 JPEG 压缩
    Input: img (H, W, C) range [0, 1], RGB
    Output: img_lq (H, W, C) range [0, 1], RGB
    """
    img_bgr = (img * 255.0).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img_bgr, encode_param)
    img_lq_bgr = cv2.imdecode(encimg, 1)
    img_lq = cv2.cvtColor(img_lq_bgr, cv2.COLOR_BGR2RGB)
    img_lq = img_lq.astype(np.float32) / 255.0
    return img_lq

# ==========================================
#  多模态数据集类 (Multi-Modal Dataset Class)
# ==========================================

class MultiModal_Dataset(data.Dataset):
    """
    支持多模态图像加载的类，并额外支持对指定模态的 JPEG 压缩处理。
    """
    def __init__(self, root_dir, modalities=['rgb', 'nir'], patch_size=128, augment=True, is_train=True,
                 jpeg_compress_modalities=None, quality_min=30, quality_max=80):
        """
        Args:
            root_dir (str): 数据集根目录
            modalities (list): 模态名称列表
            patch_size (int): 图像裁剪大小
            augment (bool): 是否进行数据增强
            is_train (bool): 是否为训练模式
            jpeg_compress_modalities (list): 指定需要进行 JPEG 压缩的模态列表
            quality_min (int): 压缩质量的最低值
            quality_max (int): 压缩质量的最高值
        """
        super(MultiModal_Dataset, self).__init__()
        self.patch_size = patch_size
        self.augment = augment
        self.is_train = is_train
        self.modalities = modalities
        
        # JPEG 压缩相关设置
        self.jpeg_compress_modalities = jpeg_compress_modalities or []
        for modality in self.jpeg_compress_modalities:
            if modality not in self.modalities:
                raise ValueError(f"JPEG compress modality '{modality}' not found in modalities: {self.modalities}")
        self.quality_min = quality_min
        self.quality_max = quality_max

        # 根据训练/测试模式确定数据路径
        split_folder = 'train' if is_train else 'test'
        self.data_root = os.path.join(root_dir, split_folder)
        
        if not os.path.isdir(self.data_root):
            raise FileNotFoundError(f"Data folder not found at: {self.data_root}")

        # 查找并匹配所有模态的图像路径
        self.image_paths = self._find_and_match_files()
        if not self.image_paths:
            raise ValueError(f"No matching image pairs found in {self.data_root} for modalities {self.modalities}")

    def _find_and_match_files(self):
        paths_dict = defaultdict(dict)
        ext_list = ['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff', 'webp']
        for modality in self.modalities:
            modality_path = os.path.join(self.data_root, modality)
            if not os.path.isdir(modality_path):
                print(f"Warning: Modality folder not found: {modality_path}")
                continue
            for ext in ext_list:
                pattern = '*.' + ''.join(f'[{c}{c.upper()}]' for c in ext)
                files = glob.glob(os.path.join(modality_path, pattern))
                for file_path in files:
                    filename = os.path.basename(file_path)
                    paths_dict[filename][modality] = file_path
        
        matched_paths = []
        for filename, modality_files in sorted(paths_dict.items()):
            if len(modality_files) == len(self.modalities):
                matched_paths.append({m: modality_files[m] for m in self.modalities})
        return matched_paths

    def __getitem__(self, index):
        path_group = self.image_paths[index]
        
        # 1. 读取所有模态的原始图像 (Ground Truth)
        img_gt_list = []
        for modality in self.modalities:
            path = path_group[modality]
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4: img = img[:, :, :3]
            if img.shape[2] == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img_gt_list.append(img)

        # 2. 裁剪图像 (Cropping)
        if self.is_train:
            H, W, _ = img_gt_list[0].shape
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            img_gt_list = [img[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :] for img in img_gt_list]
            if self.augment:
                img_gt_list = augment(img_gt_list)
        elif self.patch_size is not None and self.patch_size > 0:
            H, W, _ = img_gt_list[0].shape
            ps = min(H, W, self.patch_size)
            top, left = (H - ps) // 2, (W - ps) // 2
            img_gt_list = [img[top:top + ps, left:left + ps, :] for img in img_gt_list]
        
        # 3. 创建返回字典，包含所有GT图像
        return_dict = {f"{m}_gt": t for m, t in zip(self.modalities, np2tensor(img_gt_list))}

        # 4. 对指定模态进行JPEG压缩，生成LQ输入
        for modality in self.jpeg_compress_modalities:
            modality_idx = self.modalities.index(modality)
            img_to_compress = img_gt_list[modality_idx]
            
            quality = random.randint(self.quality_min, self.quality_max) if self.is_train else self.quality_max
            img_lq = jpeg_compress(img_to_compress, quality)
            
            # 将LQ图像添加到返回字典
            return_dict[f"{modality}_lq"] = np2tensor([img_lq])[0]

        return return_dict

    def __len__(self):
        return len(self.image_paths)

# ==========================================
#  使用示例 (Demo)
# ==========================================
if __name__ == '__main__':
    database_root = os.path.expanduser('~/database/RGB-NIR')
    modalities_to_load = ['rgb', 'nir']
    
    print("--- Testing Multi-Modal Loader with JPEG compression ---")
    try:
        jpeg_dataset = MultiModal_Dataset(
            root_dir=database_root,
            modalities=modalities_to_load,
            patch_size=128,
            is_train=True,
            jpeg_compress_modalities=['rgb', 'nir'], # 对 'rgb' 和 'nir' 模态进行压缩
            quality_min=20,
            quality_max=70
        )
        
        if len(jpeg_dataset) > 0:
            print(f"Found {len(jpeg_dataset)} training pairs.")
            first_item = jpeg_dataset[0]
            print("Keys in a dataset item:", first_item.keys())
            # 预期输出: ['rgb_gt', 'nir_gt', 'rgb_lq', 'nir_lq']
            
            for key, value in first_item.items():
                print(f"{key}: {value.shape}")

    except (FileNotFoundError, ValueError) as e:
        print(f"\nError: {e}")
