import warnings

warnings.filterwarnings("ignore")
import sys

sys.path.append("3rdparty/pats")
import argparse
import os
import random

import cv2
import mmengine
import numpy as np
import torch
import yaml
from models.pats import PATS
from torch.cuda.amp import GradScaler as GradScaler
from torch.cuda.amp import autocast as autocast
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from utils.utils import Resize_img

from vlm_grounder.utils import SceneInfoHandler, get_scene_ids_from_csv


class Scannet(Dataset):
    def __init__(
        self,
        data_path="data/scannet/posed_images",
        scene_infos="data/scannet/scannet_instance_data/scenes_train_val_info_w_images.pkl",
        scene_ids=None,
    ):
        self.data_path = data_path
        self.scene_infos = SceneInfoHandler(scene_infos)

        if scene_ids is None:
            self.scene_ids = ["scene0011_00"]
        else:
            self.scene_ids = scene_ids  # sorted

        self.all_pairs = (
            self._get_all_pairs()
        )  # list of pairs, each pair is a dict with keys 'image0' and 'image1'

    def _get_all_pairs(self):
        all_pairs = []
        for scene_id in self.scene_ids:
            num_posed_images = self.scene_infos.get_num_posed_images(scene_id)
            all_images_files = [
                f"{scene_id}/{i:05d}.jpg" for i in range(num_posed_images)
            ]

            # construct a pair, each pair is a dict with keys 'image0' and 'image1'
            # each two in the all_images_files should form a pair
            # so the final pairs will be n * (n + 1) / 2
            for i in range(len(all_images_files)):
                for j in range(i + 1, len(all_images_files)):
                    pair = {
                        "image0": all_images_files[i],
                        "image1": all_images_files[j],
                    }
                    all_pairs.append(pair)

        return all_pairs

    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, index):
        pair = self.all_pairs[index]
        left_rgb_path = os.path.join(self.data_path, pair["image0"])
        right_rgb_path = os.path.join(self.data_path, pair["image1"])

        size = 640  # the max size, the image will become 480 * 640 (h, w)

        left = cv2.imread(left_rgb_path)[:, :, [2, 1, 0]]
        ori_image_shape = left.shape[:2]
        h_l, w_l = left.shape[:2]
        max_shape_l = max(h_l, w_l)
        size_l = size / max_shape_l  # 640 / 1296

        # although Resize_img will crop the image to keep aspect
        # but current the target shape has the same aspect ratio as the original image
        # so no crop is conducted
        # however, the h_l * size_l < 480, so will be padded
        # and the scale factor is the same of both H and W
        left = Resize_img(left, np.array([int(w_l * size_l), int(h_l * size_l)]))
        h_l2, w_l2 = left.shape[:2]
        assert (
            h_l2 <= 480 and w_l2 == 640
        ), f"ScanNet image should be smaller than 480 * 640 after scaling: {pair}."

        right = cv2.imread(right_rgb_path)[:, :, [2, 1, 0]]
        h_r, w_r = right.shape[:2]
        max_shape_r = max(h_r, w_r)
        size_r = size / max_shape_r
        right = Resize_img(right, np.array([int(w_r * size_r), int(h_r * size_r)]))
        h_r2, w_r2 = right.shape[:2]
        assert (
            size_r == size_l
        ), f"The scale factor should be the same for the two images: {pair}."

        left = cv2.copyMakeBorder(
            left, 0, 480 - h_l2, 0, 640 - w_l2, cv2.BORDER_CONSTANT, None, 0
        )
        right = cv2.copyMakeBorder(
            right, 0, 480 - h_r2, 0, 640 - w_r2, cv2.BORDER_CONSTANT, None, 0
        )

        data = {
            "image0_name": pair["image0"],
            "image0": left,
            "image1_name": pair["image1"],
            "image1": right,
            "scale_factor": size_l,
            "ori_image_shape": ori_image_shape,  # * (H, W) This is used to check if matching out of the image
        }

        return data


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="3rdparty/pats/configs/test_scannet.yaml"
    )
    parser.add_argument("--scale_factor", type=float, default=1.0)  # * no used
    parser.add_argument(
        "--vg_file",
        type=str,
        default="data/scannet/grounding/referit3d/scanrefer_sampled_250_relations.csv",
    )  # * no used

    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, "r") as f:
            yaml_dict = yaml.safe_load(f)
            for k, v in yaml_dict.items():
                args.__dict__[k] = v

    # initialize random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model = PATS(args)
    model.config.checkpoint = os.path.join("3rdparty/pats", model.config.checkpoint)
    model.config.checkpoint2 = os.path.join("3rdparty/pats", model.config.checkpoint2)
    model.config.checkpoint3 = os.path.join("3rdparty/pats", model.config.checkpoint3)
    model.load_state_dict()
    model = model.cuda().eval()

    vg_file_path = args.vg_file
    scene_ids = get_scene_ids_from_csv(vg_file_path)
    # sort
    scene_ids.sort()

    output_dir = "data/scannet/scannet_match_data"
    # output_dir = f"{output_dir}/{os.path.basename(vg_file_path).replace('.csv', '')}"
    print(f"Output dir is {output_dir}")
    mmengine.mkdir_or_exist(output_dir)

    scannet_dataset = Scannet(scene_ids=scene_ids)
    scannet_loader = DataLoader(
        scannet_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )

    pkl_file_path = f"{output_dir}/{os.path.basename(vg_file_path).replace('.csv', '')}_exhaustive_matching.pkl"
    if os.path.exists(pkl_file_path):
        matching_results = mmengine.load(pkl_file_path)
    else:
        matching_results = {}

    print(f"Starting processing {len(scene_ids)} scene_ids")
    index = 0
    for data in tqdm(scannet_loader):
        data["image0"] = data["image0"].cuda()
        scene_id, image0_id = data["image0_name"][0].split("/")
        scene_id, image1_id = data["image1_name"][0].split("/")
        image0_id = image0_id.split(".")[0]
        image1_id = image1_id.split(".")[0]
        assert image0_id < image1_id  # names are sorted
        if scene_id in matching_results:
            if (image0_id, image1_id) in matching_results[scene_id]:
                print(
                    f"[ExhaustiveMatching] {scene_id}:{(image0_id, image1_id)} already processed"
                )
                continue
        else:
            matching_results[scene_id] = {}
            matching_results[scene_id]["processed_num"] = 0

        data["image1"] = data["image1"].cuda()
        scale_factor = data["scale_factor"].cuda()  # tensor
        ori_h, ori_w = data["ori_image_shape"]  # H, W, tensor
        ori_h, ori_w = ori_h.cuda(), ori_w.cuda()

        with torch.no_grad():
            result = model(data)

        # change this to conduct on gpu/tensor
        kp0 = result["matches_l"]
        kp1 = result["matches_r"]
        if len(kp0) == 0 or len(kp1) == 0:
            # no mathcing, kp0 shape is 0 * 2
            matching_results[scene_id]["processed_num"] += 1
            continue

        kp0 /= scale_factor
        kp0 = torch.round(kp0).to(torch.int16)
        kp1 /= scale_factor
        kp1 = torch.round(kp1).to(torch.int16)

        # filter out those out of range
        mask1 = torch.logical_and(
            torch.logical_and(kp0[:, 1] >= 0, kp0[:, 1] < ori_w),
            torch.logical_and(kp0[:, 0] >= 0, kp0[:, 0] < ori_h),
        )

        mask2 = torch.logical_and(
            torch.logical_and(kp1[:, 1] >= 0, kp1[:, 1] < ori_w),
            torch.logical_and(kp1[:, 0] >= 0, kp1[:, 0] < ori_h),
        )
        # 结合 mask1 和 mask2
        mask = torch.logical_and(mask1, mask2)

        kp0 = kp0[mask]
        kp1 = kp1[mask]

        # save the keypoint
        matching_results[scene_id][(image0_id, image1_id)] = {
            "kp0": kp0.cpu().numpy(),
            "kp1": kp1.cpu().numpy(),
        }
        matching_results[scene_id]["processed_num"] += 1

        index += 1
        if index % 50 == 0:
            mmengine.dump(matching_results, pkl_file_path)

    # save results
    mmengine.dump(matching_results, pkl_file_path)
