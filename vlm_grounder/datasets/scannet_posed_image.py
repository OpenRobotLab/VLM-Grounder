import os

import cv2
import mmengine
import torch
from torch.utils.data import DataLoader, Dataset


class ScanNetPosedImages(Dataset):
    def __init__(
        self,
        data_root,
        info_file,
        scene_id_file=None,
        scene_id_list=None,
        return_image=False,
    ):
        # 存储数据根目录和信息文件路径
        self.data_root = data_root
        self.info_file = info_file
        self.return_image = return_image

        # 读取场景信息
        self.scene_infos = mmengine.load(info_file)
        if scene_id_list is not None:
            self.scene_ids = scene_id_list
        elif scene_id_file is not None:
            self.scene_ids = mmengine.list_from_file(scene_id_file)
        else:
            self.scene_ids = []
        print(f"[Dataset]: Total number of scenes: {len(self.scene_ids)}.")

        # 构建索引映射
        self.index_mapping = []
        for scene_id in self.scene_ids:
            scene_info = self.scene_infos[scene_id]
            num_posed_images = scene_info["num_posed_images"]
            for i in range(num_posed_images):
                image_id = f"{i:05d}"
                self.index_mapping.append((scene_id, image_id))

        # * print the total image number
        print(f"[Dataset]: Total number of images: {len(self)}.")

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        scene_id, image_id = self.index_mapping[idx]
        scene_info = self.scene_infos[scene_id]
        image_info = scene_info["images_info"][image_id]

        # 加载图片（这里只提供路径，实际加载需自行完成）
        # * paths are like 'data/scannet/posed_images/scene0706_00/00000.jpg'
        image_path = os.path.join(self.data_root, image_info["image_path"])
        # image = read_image(image_path)  # 假设图片是RGB格式

        # 获取其他信息
        extrinsic_matrix = image_info["extrinsic_matrix"]

        data_sample = {
            "image_path": image_path,  # 'image': image,
            "scene_id": scene_id,
            "extrinsic_matrix": extrinsic_matrix,
        }

        if self.return_image:
            image_bgr_np = cv2.imread(image_path)  # * H, W, 3
            data_sample["image_bgr"] = image_bgr_np

        # 返回所需信息，可根据需求自行调整
        return data_sample


if __name__ == "__main__":
    # 使用示例
    # 假定数据集和信息文件已经准备好
    data_root = ""
    info_file = "data/scannet/scannet_instance_data/scenes_train_val_info_w_images.pkl"
    scene_id_file = "data/scannet/meta_data/scannetv2_val.txt"
    dataset = ScanNetPosedImages(data_root, info_file, scene_id_file, return_image=True)
    # dataloader
    dataloader = DataLoader(dataset=dataset, batch_size=2)

    # 访问第一个元素
    for sample in dataloader:
        print(sample)
        break
