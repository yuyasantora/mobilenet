import json

input_json_path = "dataset/quadrant-enumeration-disease/train.json" # ★★★ 元のファイルパス
output_json_path = "dataset/quadrant-enumeration-disease/train_corrected.json" # ★★★ 修正後のファイルパス

try:
    with open(input_json_path, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"エラー: ファイルが見つかりません: {input_json_path}")
    exit()
except json.JSONDecodeError:
    print(f"エラー: JSONのデコードに失敗しました。ファイル形式を確認してください: {input_json_path}")
    exit()

if 'annotations' not in data:
    print("エラー: 'annotations' キーがJSONデータ内に見つかりません。")
    exit()

modified_count = 0
for annotation in data['annotations']:
    if 'category_id_3' in annotation:
        annotation['category_id'] = annotation.pop('category_id_3') # キー名を変更し、値をコピー
        modified_count += 1

if modified_count > 0:
    try:
        with open(output_json_path, 'w') as f:
            json.dump(data, f, indent=4) # indentは見やすいように整形する場合
        print(f"修正完了。 {modified_count} 個のアノテーションで 'category_id_3' を 'category_id' に変更しました。")
        print(f"修正後のファイル: {output_json_path}")
    except IOError:
        print(f"エラー: ファイルへの書き込みに失敗しました: {output_json_path}")
else:
    print("'category_id_3' を持つアノテーションは見つかりませんでした。ファイルは変更されていません。")
