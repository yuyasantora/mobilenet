import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.datasets import CocoDetection
import torchvision.transforms as T

class CustomCocoDetection(CocoDetection): # torchvision.datasets.CocoDetection を継承していると仮定
        def __init__(self, root, annFile, transform=None, target_transform=None, target_size=(320, 320)): # target_size を追加
            super().__init__(root, annFile) # CocoDetectionの初期化方法に合わせる
            self.transform = transform
            self.target_transform = target_transform # 通常、物体検出ではあまり使わない
            self.target_h, self.target_w = target_size

        def __getitem__(self, idx):
            coco = self.coco
            img_id = self.ids[idx]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            target_coco_format = coco.loadAnns(ann_ids)

            path = coco.loadImgs(img_id)[0]['file_name']
            img_pil = Image.open(os.path.join(self.root, path)).convert('RGB')
            
            original_w, original_h = img_pil.size

            boxes = []
            labels = []
            areas = []
            iscrowd = []
            valid_annotations = False # 有効なアノテーションがあるかどうかのフラグ

            for t in target_coco_format:
                if t.get('ignore', 0) == 1 or t.get('iscrowd', 0) == 1: # 'ignore'属性も考慮
                    continue
                
                # COCO bbox format: [x_min, y_min, width, height]
                x_min, y_min, w, h = t['bbox']

                # ゼロ幅またはゼロ高さのボックスはスキップ (またはエラー処理)
                if w <= 0 or h <= 0:
                    continue

                # PyTorch format: [x_min, y_min, x_max, y_max]
                boxes.append([x_min, y_min, x_min + w, y_min + h])
                labels.append(t['category_id']) # category_id は事前に1から始まるように調整されているか確認
                areas.append(t['area'])
                iscrowd.append(t['iscrowd']) # iscrowdは0のはず
                valid_annotations = True

            target = {}
            if valid_annotations:
                target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
                target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
                # スケーリング比を計算
                scale_w = self.target_w / original_w
                scale_h = self.target_h / original_h

                # バウンディングボックスをスケーリング
                target["boxes"][:, 0] = target["boxes"][:, 0] * scale_w # x_min
                target["boxes"][:, 1] = target["boxes"][:, 1] * scale_h # y_min
                target["boxes"][:, 2] = target["boxes"][:, 2] * scale_w # x_max
                target["boxes"][:, 3] = target["boxes"][:, 3] * scale_h # y_max

                # スケーリング後のボックスが画像の範囲内に収まるようにクリッピング (オプションだが推奨)
                target["boxes"][:, 0::2] = target["boxes"][:, 0::2].clamp(min=0, max=self.target_w) # x座標
                target["boxes"][:, 1::2] = target["boxes"][:, 1::2].clamp(min=0, max=self.target_h) # y座標
                
                # 面積もスケーリング（通常、損失計算には直接使われないが、参考として）
                # scaled_areas = []
                # for box_scaled in target["boxes"]:
                #     sw = box_scaled[2] - box_scaled[0]
                #     sh = box_scaled[3] - box_scaled[1]
                #     scaled_areas.append(sw * sh)
                # target["area"] = torch.as_tensor(scaled_areas, dtype=torch.float32)

            else: # 有効なアノテーションが一つもなかった場合
                target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                target["labels"] = torch.zeros((0,), dtype=torch.int64) # 空のラベルテンソル
                # target["area"] = torch.zeros((0,), dtype=torch.float32)


            target["image_id"] = torch.tensor([img_id])
            # iscrowd は有効なアノテーションがない場合、空にするか、
            # もしモデルが特定の形式を期待するならそれに合わせる。
            target["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.int64) if valid_annotations else torch.zeros((0,), dtype=torch.int64)
            if valid_annotations: # areaも有効なアノテーションがある場合のみ設定
                 target["area"] = torch.as_tensor(areas, dtype=torch.float32) * (scale_w * scale_h) # 面積もスケーリング
            else:
                 target["area"] = torch.zeros((0,), dtype=torch.float32)


            # PILイメージをターゲットサイズにリサイズ
            img_pil_resized = img_pil.resize((self.target_w, self.target_h), Image.BILINEAR) # または他の補間方法

            if self.transform is not None:
                img_tensor = self.transform(img_pil_resized) # ToTensor, Normalizeなど
            else: # transformがNoneの場合でもテンソルには変換する必要があることが多い
                img_tensor = T.ToTensor()(img_pil_resized)


            # target_transform があれば適用 (通常は物体検出ではあまり使わない)
            # if self.target_transform is not None:
            #     target = self.target_transform(target)

            return img_tensor, target
        
    
    
