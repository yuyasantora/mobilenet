import torch
import argparse
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_resnet50_fpn
from torchvision import transforms as T
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import tqdm

from data import CustomCocoDetection # training scriptで使ったdata.pyをインポート

def evaluate(model, dataloader, device):
    """モデルのmAPを計算する関数"""
    model.to(device)
    model.eval()

    # mAP計算用のメトリックを初期化
    metric = MeanAveragePrecision(box_format='xyxy').to(device)

    # ★★★ デバッグ用のフラグを追加 ★★★
    debug_printed = False 

    with torch.no_grad():
        for images, targets in tqdm.tqdm(dataloader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            
            # モデルの推論結果を取得
            preds = model(images)
            
            # ★★★ デバッグコードを追加 ★★★
            if not debug_printed:
                print("\n--- Raw Prediction Output (first batch) ---")
                print(f"Number of predictions: {len(preds)}")
                if len(preds) > 0:
                    first_pred = preds[0]
                    print("Prediction for the first image:")
                    for k, v in first_pred.items():
                        print(f"  '{k}': shape={v.shape}, dtype={v.dtype}, device={v.device}")
                        # 実際の値も少し表示
                        if v.numel() > 0:
                            print(f"     values: {v[:5].tolist()}") # 先頭5件まで表示
                        else:
                            print("     (empty tensor)")
                print("--- End of Raw Prediction ---\n")
                debug_printed = True # 一度だけ表示する

            # 正解ラベルの形式をメトリックに合わせて整形
            # CustomCocoDetectionが返す形式が {'boxes': ..., 'labels': ...} の辞書であると想定
            targets_for_metric = []
            for t in targets:
                targets_for_metric.append({
                    'boxes': t['boxes'].to(device),
                    'labels': t['labels'].to(device)
                })

            # 1バッチ分の予測と正解をメトリックに渡す
            metric.update(preds, targets_for_metric)

    # 全データの評価結果を集計
    results = metric.compute()
    return results

def main():
    parser = argparse.ArgumentParser(description="Object Detection Model Evaluation")
    parser.add_argument("--model_type", type=str, required=True, choices=['teacher', 'student'], help=" 'teacher' or 'student' ")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to model weights (.pth file)")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to the root of the test dataset")
    parser.add_argument("--ann_file", type=str, required=True, help="Path to the annotation file for the test dataset")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # --- モデルの読み込み ---
    if args.model_type == 'teacher':
        model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=5)
        # 学習済み重みを読み込む
        model.load_state_dict(torch.load(args.weights_path, map_location=args.device))
    else: # student
        # ★★★ 修正点 ★★★
        # 生徒モデルを定義する際は、学習時と同じように教師モデルの構造を一部利用する
        # ダミーの教師モデルをインスタンス化して、その一部を生徒モデルに渡す
        teacher_for_init = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=5)
        model = fasterrcnn_mobilenet_v3_large_fpn(
            pretrained=False, 
            num_classes=5, 
            box_roi_pool=teacher_for_init.roi_heads.box_roi_pool
        )
        # 学習済みの生徒モデルの重みを読み込む
        model.load_state_dict(torch.load(args.weights_path, map_location=args.device))
    
    print(f"Loaded {args.model_type} model from {args.weights_path}")

    # --- データセットの準備 ---
    transform = T.Compose([T.ToTensor()]) # 評価時はNormalizeは不要なことが多い
    target_size = (320, 320) # 学習時と同じサイズ
    
    # 評価用データセット（shuffle=Falseに注意）
    dataset = CustomCocoDetection(root=args.dataset_root, annFile=args.ann_file, transform=transform, target_size=target_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # --- 評価の実行 ---
    results = evaluate(model, dataloader, args.device)

    # --- 結果の表示 ---
    print("\n--- Evaluation Results ---")
    for k, v in results.items():
        # ★★★ 修正点 ★★★
        # vがスカラー（要素が1つ）の場合のみ .item() を使う
        if v.numel() == 1:
            print(f"{k}: {v.item():.4f}")
        else:
            # それ以外の場合は、テンソルをそのまま表示する
            print(f"{k}: {v}")
    print("--------------------------")

if __name__ == "__main__":
    main() 