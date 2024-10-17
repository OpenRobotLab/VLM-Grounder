# * converting
import argparse
import csv

import mmengine

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_json_path",
    type=str,
    default="data/scannet/grounding/scanrefer/ScanRefer_filtered_val.json",
)
parser.add_argument(
    "--output_csv_path",
    type=str,
    default="data/scannet/grounding/referit3d/scanrefer_val.csv",
)  # * use this or --max-images-per-scene
parser.add_argument("--nproc", type=int, default=6)
args = parser.parse_args()

input_json_path = args.input_json_path
output_csv_path = args.output_csv_path
# 读取JSON文件
data = mmengine.load(input_json_path)
mmengine.mkdir_or_exist("data/scannet/grounding/referit3d")

# 创建CSV文件并写入数据
with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
    fieldnames = ["scan_id", "target_id", "instance_type", "utterance", "tokens"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    for item in data:
        writer.writerow(
            {
                "scan_id": item["scene_id"],
                "target_id": item["object_id"],
                "instance_type": item["object_name"],
                "utterance": item["description"],
                "tokens": item["token"],
            }
        )

print("JSON to CSV conversion completed successfully.")
