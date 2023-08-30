import os
import numpy as np
import torch
from PIL import Image

class PennFundanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
    
    def __getitem__(self, idx):
        # 이미지와 마스크 읽어오기
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx]) 
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx]) 
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        maks = np.array(mask)
        obj_ids = np.unique(mask)
        # 첫번째 id는 배경이기 때문에 제거
        obj_ids = obj_ids[1:]
        
        # 컬러 인코딩된 마스크를 바이너리 마스크 세트로 나눔
        masks = mask == obj_ids[:, None, None]
        
        # 각 마스크의 바운딩 박스 좌표를 얻음
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        
        # torch.Tensor 타입으로 변환
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # 객체 종류는 한 종류(사람)만 존재
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # 모든 인스턴스는 군중(crowd) 상태가 아님을 가정합니다
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target
    
    def __len__(self):
        return len(self.imgs)