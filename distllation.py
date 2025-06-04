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
parser.add_argument("--save_dir", type=str, default="C:/Users/ohhara/mobilenetv2-ssd/distillation/student_model")
parser.add_argument("--alpha", type=float, default=0.5) # distillation loss weight

args = parser.parse_args()

# それぞれのモデルの読み込み
teacher_model = ssd300_vgg16(pretrained=False, num_classes=5)
teacher_model.load_state_dict(torch.load("C:/Users/ohhara/mobilenetv2-ssd/distillation/teacher_model/model_epoch_200.pth"))
teacher_model.to(args.device)
student_model = ssdlite320_mobilenet_v3_large(pretrained=False, num_classes=4)
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

# 蒸留損失の計算
def distillation_loss(teacher_model_outputs_list, student_model_outputs_list):
    T = 1.0

    print("Debug: Teacher output list length:", len(teacher_model_outputs_list))
    if teacher_model_outputs_list:
        print("Debug: Shape of first teacher scores:", teacher_model_outputs_list[0]['scores'].shape)

    print("Debug: Student output list length:", len(student_model_outputs_list))
    if student_model_outputs_list:
        print("Debug: Shape of first student scores:", student_model_outputs_list[0]['scores'].shape)

    teacher_scores_all = []
    for i, out in enumerate(teacher_model_outputs_list):
        print(f"Debug: Teacher scores for image {i} shape: {out['scores'].shape}")
        teacher_scores_all.append(out['scores'])
    
    student_scores_all = []
    for i, out in enumerate(student_model_outputs_list):
        print(f"Debug: Student scores for image {i} shape: {out['scores'].shape}")
        student_scores_all.append(out['scores'])

    try:
        teacher_scores_cat = torch.cat(teacher_scores_all, dim=0)
        student_scores_cat = torch.cat(student_scores_all, dim=0)
        print("Debug: Shape of concatenated teacher_scores_cat:", teacher_scores_cat.shape)
        print("Debug: Shape of concatenated student_scores_cat:", student_scores_cat.shape)
    except RuntimeError as e:
        print(f"Error during score concatenation: {e}")
        print(f"Teacher scores shapes: {[s.shape for s in teacher_scores_all]}")
        print(f"Student scores shapes: {[s.shape for s in student_scores_all]}")
        raise e

    log_p_student = F.log_softmax(student_scores_cat / T, dim=1)
    q_teacher = F.softmax(teacher_scores_cat / T, dim=1)

    if log_p_student.shape[1] != q_teacher.shape[1]:
        print(f"Critical Error: Class mismatch for KL divergence. Student has {log_p_student.shape[1]} classes, teacher has {q_teacher.shape[1]} classes.")
        print("Please align the number of classes for teacher and student models for distillation to work.")
        return torch.tensor(0.0, device=log_p_student.device, requires_grad=True) # ダミー損失

    distillation = F.kl_div(log_p_student, q_teacher, reduction='batchmean', log_target=False) * (T * T)
    return distillation

# <--- 生徒モデルの学習 --->

for epoch in range(args.epochs):
    student_model.train()
    teacher_model.eval()
    epoch_loss = 0
    writer = SummaryWriter(log_dir=args.save_dir)
    for images, targets in tqdm.tqdm(dataloader):
        images_dev = [image.to(args.device) for image in images]
        targets_dev = [{k: v.to(args.device) for k, v in t.items()} for t in targets]
       

        # 勾配のリセット
        optimizer.zero_grad()
        task_loss_dict = student_model(images_dev, targets_dev)
        task_loss = sum(loss for loss in task_loss_dict.values())

        with torch.no_grad():
            teacher_model_outputs_list = teacher_model(images_dev)

        student_model.eval()
        with torch.no_grad():
            student_model_outputs_list_for_distill = student_model(images_dev)
        student_model.train()

        # 蒸留損失の計算
        distill_loss_val = distillation_loss(teacher_model_outputs_list, student_model_outputs_list_for_distill)

        loss = task_loss + args.alpha * distill_loss_val
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_epoch_loss = epoch_loss / len(dataloader)
    writer.add_scalar('loss', avg_epoch_loss, epoch)
    print(f"epoch:{epoch + 1}")
    print(f"loss:{avg_epoch_loss}")










