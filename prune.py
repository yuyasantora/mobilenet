import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.onnx import export
from  torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn

import torch.nn.utils.prune as prune
import time
# 今回使うモデルの読み込み
model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=5)
model.load_state_dict(torch.load("C:/Users/ohhara/mobilenetv2-ssd/distillation/teacher_model/model_epoch_190.pth"))

# モデルをonnxファイルに変換する関数を定義
def model_to_onnx(model, output_file, input_shape=(1, 1, 224, 224)):
    model.eval()
    dummy_input = torch.randn(input_shape)
    input_names = ["input"]
    output_names = ["output"]

    export(model, dummy_input, output_file, input_names=input_names, output_names=output_names)

    return output_file

model_to_onnx(model, "fasterrcnn_resnet50_fpn.onnx")


"""普通に推論させてみる(デモ
タ)"""
input_size = (1, 3, 224, 224)
output = model(input_size) # ウォームアップ

start_time = time.time()

result = model(input_size)

end_time = time.time()

print(f"推論時間: {end_time - start_time}秒")

"""ようやく枝刈りの実装"""
# 全ての畳み込み層のパラメータを対象にする
parameters_to_prune = [conv for conv in model.features.parameters() if isinstance(conv, nn.Conv2d)]

# 大域的、非構造、強度枝刈り
prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.7)



