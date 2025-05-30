from torchvision.models.detection import ssd300_vgg16, ssdlite320_mobilenet_v3_large
from torchvision import transforms as T
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
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=0.0001)
parser.add_argument("--save_dir", type=str, default="C:/Users/ohhara/mobilenetv2-ssd/distillation/teacher_model")

# それぞれのモデルの読み込み
teacher_model = ssd300_vgg16(pretrained=False, num_classes=4)
student_model = ssdlite320_mobilenet_v3_large(pretrained=False, num_classes=4)

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

# 蒸留損失の計算
def distillation_loss(teacher_outputs, student_outputs, alpha=0.5):
    T = 1.0
    p = F.softmax(student_outputs['scores'] / T, dim=1)
    q = F.softmax(teacher_outputs['scores'] / T, dim=1)
    distillation = F.kl_div(p, q, reduction='batchmean') * (T * T) + F.kl_div(p, q, reduction='batchmean')
    return distillation

# <--- 生徒モデルの学習 --->

for epoch in range(parser.epochs):
    student_model.train()
    teacher_model.eval()
    epoch_loss = 0
    writer = SummaryWriter(log_dir=parser.save_dir)
    for images, targets in tqdm.tqdm(dataloader):
        images = [image.to(parser.device) for image in images]
        targets = [{k: v.to(parser.device) for k, v in t.items()} for t in targets]

        # 勾配のリセット
        parser.optimizer.zero_grad()
        task_loss_dict = student_model(images, targets)
        task_loss = sum(loss for loss in task_loss_dict.values())

        with torch.no_grad():
            teacher_outputs = teacher_model(images)
            teacher_outputs = [{k: v.to(parser.device) for k, v in t.items()} for t in teacher_outputs]

        student_outputs = student_model(images)
        student_outputs = [{k: v.to(parser.device) for k, v in t.items()} for t in student_outputs]

        # 蒸留損失の計算
        distillation = distillation_loss(teacher_outputs, student_outputs)

        loss = task_loss + parser.alpha * distillation
        loss.backward()
        parser.optimizer.step()
        epoch_loss += loss.item()
    
    avg_epoch_loss = epoch_loss / len(dataloader)
    writer.add_scalar('loss', avg_epoch_loss, epoch)
    print(f"epoch:{epoch + 1}")
    print(f"loss:{avg_epoch_loss}")










