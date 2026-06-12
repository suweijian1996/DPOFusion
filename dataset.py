import json
import os
import cv2
import numpy as np
import random

from PIL import Image
from torch.utils.data import Dataset


def _normalize(img_np):
    return (img_np.astype(np.float32) / 127.5) - 1.0

def _normalize_01(img_np):
    return (img_np.astype(np.float32) / 255.0)

def _d4_transform(img_np, code:int):
    """
    code ∈ {0..7}
      0..3: 旋转 k*90°
      4..7: 先旋转 k*90° 再水平翻转
    使用 np.rot90/np.flip，避免插值。
    """
    k = code % 4
    out = np.rot90(img_np, k, axes=(0, 1))  # (H,W,C)
    if code >= 4:
        out = np.flip(out, axis=1)          # 水平翻转
    return out.copy()                        # 保证是连续内存

class MyDataset(Dataset):
    def __init__(self, json_path="./data/train/MSRS/annotations.json",
                 root="./data/train/MSRS",
                 size=256, random_crop=False, task='fusion', augment=True):
        self.task = task  # 'fusion' or 'OD' or 'Sig'
        self.data = []
        self.root = root
        self.size = size
        self.random_crop = random_crop
        self.augment = augment
        with open(json_path, 'rt') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def _process_one(self, path, crop_coords=None):
        img = Image.open(path).convert("L")  # 灰度图
        img_np = np.array(img).astype(np.uint8)
        img_np = np.stack([img_np] * 3, axis=-1)  # 灰度转伪RGB

        h, w = img_np.shape[:2]
        if crop_coords is None:
            crop_size = min(h, w)
            start_h = (h - crop_size) // 2
            start_w = (w - crop_size) // 2
        else:
            start_h, start_w, crop_size = crop_coords

        img_np = img_np[start_h:start_h + crop_size, start_w:start_w + crop_size]
        img_np = cv2.resize(img_np, (self.size, self.size), interpolation=cv2.INTER_AREA)

        return _normalize(img_np)
    
    def _process_blur_from_target(self, path, crop_coords=None, ksize=15, sigma=3.0):
        img = Image.open(path).convert("L")
        img_np = np.array(img).astype(np.uint8)
        img_np = np.stack([img_np] * 3, axis=-1)

        h, w = img_np.shape[:2]
        if crop_coords is None:
            crop_size = min(h, w)
            start_h = (h - crop_size) // 2
            start_w = (w - crop_size) // 2
        else:
            start_h, start_w, crop_size = crop_coords

        img_np = img_np[start_h:start_h + crop_size, start_w:start_w + crop_size]
        img_np = cv2.resize(img_np, (self.size, self.size), interpolation=cv2.INTER_AREA)

        if ksize % 2 == 0:
            ksize += 1
        ksize = max(3, ksize)

        img_np = cv2.GaussianBlur(img_np, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
        return _normalize(img_np)

    def _process_one_01(self, path, crop_coords=None):
        img = Image.open(path).convert("L")
        img_np = np.array(img).astype(np.uint8)
        img_np = np.stack([img_np] * 3, axis=-1)  # 如果你的下游期望单通道，可以去掉这行

        h, w = img_np.shape[:2]
        if crop_coords is None:
            crop_size = min(h, w)
            start_h = (h - crop_size) // 2
            start_w = (w - crop_size) // 2
        else:
            start_h, start_w, crop_size = crop_coords

        img_np = img_np[start_h:start_h + crop_size, start_w:start_w + crop_size]
        # 掩码用最近邻，避免产生非0/1的过渡值
        img_np = cv2.resize(img_np, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        return _normalize_01(img_np)

    def __getitem__(self, idx):
        item = self.data[idx]
        data = item['prompt_data']

        source_ir_filename = data['ir']
        source_vis_filename = data['vis']
        ref_filename = data['ref']
        target_filename = data['target']
        prompt = data['prompt']
        reason = data['reason']
        mask_filename = data.get('mask', None)
            
        

        ir_path = os.path.join(self.root, source_ir_filename)
        vis_path = os.path.join(self.root, source_vis_filename)
        ref_path = os.path.join(self.root, ref_filename)
        target_path = os.path.join(self.root, target_filename)
        mask_path = os.path.join(self.root, mask_filename) if mask_filename else None

        # 以 IR 图像决定裁剪窗口
        ir_img = Image.open(ir_path).convert("L")
        ir_np = np.array(ir_img)
        h, w = ir_np.shape[:2]
        crop_size = min(self.size, min(h, w))  # 防止越界

        if self.random_crop and h >= crop_size and w >= crop_size:
            start_h = random.randint(0, h - crop_size)
            start_w = random.randint(0, w - crop_size)
        else:
            start_h = (h - crop_size) // 2
            start_w = (w - crop_size) // 2

        crop_coords = (start_h, start_w, crop_size)

        # 先做裁剪+缩放+归一化
        ir_arr     = self._process_one(ir_path,     crop_coords)
        vis_arr    = self._process_one(vis_path,    crop_coords)
        tgt_arr    = self._process_one(target_path, crop_coords)
        ref_arr    = self._process_one(ref_path,    crop_coords)
        if mask_path and os.path.exists(mask_path):
            mask_arr = self._process_one_01(mask_path, crop_coords)
        else:
            mask_arr = np.zeros((self.size, self.size, 3), dtype=np.float32)

        # —— 关键：每个样本只采样一次增广 code，并对该组所有图一致应用 ——
        if self.task == 'fusion':
            code = random.randint(0, 7) if self.augment else 0  # 0..7 的 D4 变换；0 即不变
            ir_arr  = _d4_transform(ir_arr,  code)
            vis_arr = _d4_transform(vis_arr, code)
            tgt_arr = _d4_transform(tgt_arr, code)
            ref_arr = _d4_transform(ref_arr, code)
            mask_arr= _d4_transform(mask_arr,code)


        return dict(ir=ir_arr, vis=vis_arr, mask=mask_arr, target=tgt_arr, ref=ref_arr, txt=prompt, reason=reason)
