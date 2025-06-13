from torchvision.models.detection import  fasterrcnn_mobilenet_v3_large_fpn,  fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision import transforms as T
import torchvision
import tqdm
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
import torch
import argparse
from data import CustomCocoDetection
from torch.utils.data import DataLoader
import torch.nn.functional as F
# コマンドライン引数
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--epochs", type=int, default="200")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=0.0001)
parser.add_argument("--save_dir", type=str, default="C:/Users/ohhara/mobilenetv2-ssd/distillation/student_model")
parser.add_argument("--alpha", type=float, default=0.5) # distillation loss weight

args = parser.parse_args()

# それぞれのモデルの読み込み
teacher_model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=5)
teacher_model.load_state_dict(torch.load("C:/Users/ohhara/mobilenetv2-ssd/distillation/teacher_model/model_epoch_190.pth"))
teacher_model.to(args.device)
teacher_model.eval()
epoch_loss = 0
t_anchor = teacher_model.rpn.anchor_generator
anchor_gen = AnchorGenerator(t_anchor.sizes, t_anchor.aspect_ratios)
student_model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=False, num_classes=5, box_roi_pool=teacher_model.roi_heads.box_roi_pool)
student_model.rpn.box_coder.weights = teacher_model.rpn.box_coder.weights
student_model.roi_heads.box_coder.weights = teacher_model.roi_heads.box_coder.weights
student_model.rpn.nms_thresh = teacher_model.rpn.nms_thresh
student_model.to(args.device)

# Optimizer
optimizer = optim.SGD(student_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

# データセット
root = "C:/Users/ohhara/mobilenetv2-ssd/dataset/quadrant-enumeration-disease/xrays"
annFile = "C:/Users/ohhara/mobilenetv2-ssd/dataset/quadrant-enumeration-disease/train_corrected.json"
target_size = (320, 320)
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = CustomCocoDetection(root=root, annFile=annFile, transform=transform, target_size=target_size)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=lambda x: tuple(zip(*x)), drop_last=True)

def distillation_loss(teacher_logits, student_logits):
    """
    物体検出のための蒸留損失。ROIを共有することで出力を固定する。
    その後、ロジット分布間の平均二乗誤差を計算します。
    """
    

# <--- 生徒モデルの学習 --->

os.makedirs(args.save_dir, exist_ok=True)
writer = SummaryWriter(log_dir=args.save_dir)

# ハイパーパラメータの設定
conf_thr = 0.7 # teacherを信じる下限スコア
top_k = 200 # teacher枠
iou_thr = 0.5 # teacher枠とiouが重なったら捨てる

student_model.train()
teacher_model.eval()


for epoch in range(args.epochs):
    

    
    for images, targets in tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
        # 画像のリストを単一のバッチテンソルに変換
        images_stacked = torch.stack([img.to(args.device) for img in images])
        # ターゲットはリストのまま
        targets_dev = [{k: v.to(args.device) for k, v in t.items()} for t in targets]
        for t in targets_dev:
            t['boxes'] = t['boxes'].to(args.device)
            t['labels'] = t['labels'].to(args.device)
        
        # -------- 教師モデルの推論 --------
        with torch.no_grad():
            t_preds = teacher_model(images_stacked) # List[Dict]

        # -- Teacherの出力を疑似ラベルに加工、GTとマージ --
        for i in range(len(targets_dev)):
            # 疑似ラベルを作成
            t_scores = t_preds[i]['scores']
            keep = t_scores > conf_thr
            if keep.sum() > top_k: # 信頼度が高い枠がtop_kを超える場合
                top_idx = torch.topk(t_scores, top_k).indices
                mask = torch.zeros_like(t_scores, dtype=torch.bool).bool()
                mask[top_idx] = True
                keep &= mask

            t_boxes = t_preds[i]['boxes'][keep]
            t_labels = t_preds[i]['labels'][keep]

            # 既存GTとIoU > 0.7のteacher枠を捨てる
            g_boxes = targets_dev[i]['boxes']
            if g_boxes.numel() and t_boxes.numel(): # どっちも空でない
                iou = torchvision.ops.box_iou(g_boxes, t_boxes)
                keep_t = (iou.max(dim=0).values < iou_thr)
                t_boxes = t_boxes[keep_t]
                t_labels = t_labels[keep_t]

            # GTとマージ
            if t_boxes.numel():
                targets_dev[i]['boxes'] = torch.cat([g_boxes, t_boxes], dim=0)
                targets_dev[i]['labels'] = torch.cat([targets_dev[i]['labels'], t_labels], dim=0)

        
        # -- 生徒モデルの順伝播 --
        loss_dict = student_model(images_stacked, targets_dev)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ログ
        epoch_loss += loss.item()
        writer.add_scalar("loss", loss.item(), epoch)

    # 定期的にモデルを保存
    if (epoch + 1) % 10 == 0:
        save_path = os.path.join(args.save_dir, f"student_model_epoch_{epoch + 1}.pth")
        torch.save(student_model.state_dict(), save_path)
        print(f"Saved model to {save_path}")

writer.close()
print("Training finished.")










