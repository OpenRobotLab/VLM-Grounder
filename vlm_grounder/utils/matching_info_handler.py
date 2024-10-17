import os
import re

import mmengine
import torch
from mmengine.utils import Timer
from mmengine.utils.dl_utils import TimeCounter
from tqdm import tqdm

if __name__ == "__main__":
    from vlm_grounder.utils import SceneInfoHandler
else:
    from .info_handler import SceneInfoHandler

import cv2
import numpy as np
from pytorch3d.loss import chamfer_distance


class MatchingVisualizer:
    """
    Codes are directly copied from PATS
    """

    def __init__(self):
        self.color_wheel = self.make_colorwheel()

    def make_colorwheel(self):
        """
        Generates a color wheel for optical flow visualization as presented in:
            Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
            URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

        Code follows the original C++ source code of Daniel Scharstein.
        Code follows the the Matlab source code of Deqing Sun.

        Returns:
            np.ndarray: Color wheel
        """

        RY = 15
        YG = 6
        GC = 4
        CB = 11
        BM = 13
        MR = 6

        ncols = RY + YG + GC + CB + BM + MR
        colorwheel = np.zeros((ncols, 3))
        col = 0

        # RY
        colorwheel[0:RY, 0] = 255
        colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
        col = col + RY
        # YG
        colorwheel[col : col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
        colorwheel[col : col + YG, 1] = 255
        col = col + YG
        # GC
        colorwheel[col : col + GC, 1] = 255
        colorwheel[col : col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
        col = col + GC
        # CB
        colorwheel[col : col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
        colorwheel[col : col + CB, 2] = 255
        col = col + CB
        # BM
        colorwheel[col : col + BM, 2] = 255
        colorwheel[col : col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
        col = col + BM
        # MR
        colorwheel[col : col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
        colorwheel[col : col + MR, 0] = 255

        return colorwheel

    def flow_uv_to_colors(self, u, v, convert_to_bgr=False):
        """
        Applies the flow color wheel to (possibly clipped) flow components u and v.

        According to the C++ source code of Daniel Scharstein
        According to the Matlab source code of Deqing Sun

        Args:
            u (np.ndarray): Input horizontal flow of shape [H,W]
            v (np.ndarray): Input vertical flow of shape [H,W]
            convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

        Returns:
            np.ndarray: Flow visualization image of shape [H,W,3]
        """
        flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
        ncols = self.color_wheel.shape[0]
        rad = np.sqrt(np.square(u) + np.square(v))
        a = np.arctan2(-v, -u) / np.pi
        fk = (a + 1) / 2 * (ncols - 1)
        k0 = np.floor(fk).astype(np.int32)
        k1 = k0 + 1
        k1[k1 == ncols] = 0
        f = fk - k0
        for i in range(self.color_wheel.shape[1]):
            tmp = self.color_wheel[:, i]
            col0 = tmp[k0] / 255.0
            col1 = tmp[k1] / 255.0
            col = (1 - f) * col0 + f * col1
            idx = rad <= 1
            col[idx] = 1 - rad[idx] * (1 - col[idx])
            col[~idx] = col[~idx] * 0.75  # out of range
            # Note the 2-i => BGR instead of RGB
            ch_idx = 2 - i if convert_to_bgr else i
            flow_image[:, :, ch_idx] = np.floor(255 * col)
        return flow_image

    def coord_trans(self, u, v):
        rad = np.sqrt(np.square(u) + np.square(v))
        u /= rad + 1e-3
        v /= rad + 1e-3
        return u, v

    def kp_color(self, u, v, resolution):
        h, w = resolution
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x, y)
        xx, yy = self.coord_trans(xx, yy)
        vis = self.flow_uv_to_colors(xx, yy)

        color = vis[v.astype(np.int32), u.astype(np.int32)]
        return color

    def draw_kp(self, img, kps, colors):
        for i, kp in enumerate(kps):
            img = cv2.circle(
                img, (int(kp[1]), int(kp[0])), 1, colors[i].tolist(), -1
            )  # cv2 operation is inplace
        return img

    def draw_matches(self, img, kps1, kps2):
        for i, kp in enumerate(kps1):
            cv2.line(
                img,
                (int(kps1[i][1]), int(kps1[i][0])),
                (int(kps2[i][1]), int(kps2[i][0])),
                (0, 255, 0),
                1,
            )
        return img

    def vis_matches(
        self,
        image0,
        image1,
        kp0,
        kp1,
        is_draw_matches=False,
        save_vis=False,
        output_dir="outputs/intermediate_results/image_matching_visualization",
        file_name="",
        pad_width=5,
    ):
        """
        Args:
            image0/image1: numpy array with bgr format
            kp0/kp1: 2D numpy array, with kp[:, 0] refer to the height dimension(ie the y coordinate when referrting to the image, and v when using u,v coordinates).
        """
        lh, lw = image0.shape[:2]

        color = self.kp_color(kp0[:, 1], kp0[:, 0], (lh, lw))

        image0 = self.draw_kp(image0, kp0, color)
        image1 = self.draw_kp(image1, kp1, color)

        zero_image = np.zeros([lh, pad_width, 3])
        vis = np.concatenate([image0, zero_image, image1], axis=1)

        if is_draw_matches:
            new_kp1 = kp1.copy()  # This is important
            new_kp1[:, 1] += lw + pad_width
            vis = self.draw_matches(vis, kp0, new_kp1)

        if save_vis:
            mmengine.mkdir_or_exist(output_dir)
            cv2.imwrite(f"{output_dir}/{file_name}", vis)

        return vis


class MatchingInfoHandler:
    def __init__(self, info_path=None, scene_infos=None, device=None):
        if info_path is not None:
            with Timer(print_tmpl="Loading matching info file takes {:.1f} seconds."):
                self.infos = mmengine.load(info_path)
        else:
            print(f"Initializing empty matching info.")

        self.scene_infos = (
            scene_infos  # This is used for providing PosedImage information
        )
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            else:
                device = "cpu"
        self.device = device
        self.visualizer = MatchingVisualizer()

    def convert_image_id_to_key(self, image_id):
        try:
            # try to convert image_id as int
            image_id = int(image_id)

            # then convert as specific format
            image_id = f"{image_id:05d}"
        except Exception as e:
            print(f"Failed to convert image_id: {image_id}: {e}")
            return None

        return image_id

    def get_num_posed_images(self, scene_id):
        assert (
            self.scene_infos is not None
        ), "When checking completeness, the scene_info shoud not be None."
        num_posed_images = self.scene_infos.get_num_posed_images(scene_id)

        return num_posed_images

    def construct_all_possible_pairs(self, scene_id, image_id):
        num_posed_images = self.scene_infos.get_num_posed_images(scene_id)
        all_image_id_list = [i for i in range(num_posed_images)]
        # Directly create the set of all pairs
        all_pairs = self.construct_pairs_from_id_list(image_id, all_image_id_list)

        return all_pairs

    def construct_pairs_from_id_list(self, image_id, image_id_list):
        # image list should be list of int
        # image_id should be a str
        all_pairs = set(
            [
                (f"{i:05d}", image_id) if i < int(image_id) else (image_id, f"{i:05d}")
                for i in image_id_list
                if i != int(image_id)
            ]
        )

        return all_pairs

    def check_infos_completeness(self):
        print(f"Checking completeness.")
        # For each scene_info, there is a processed_num keys, which contains all the possible image pairs
        for scene_id in tqdm(self.infos):
            num_posed_images = self.get_num_posed_images(scene_id)
            total_possible_num = (
                1 + num_posed_images
            ) * num_posed_images / 2 - num_posed_images
            processed_num = self.infos[scene_id]["processed_num"]

            if processed_num != total_possible_num:
                print(
                    f"Warning: the processing number of scene_id {scene_id} does not match the total possible pairs. Processed v.s. total: {processed_num} v.s. {total_possible_num}."
                )

    def merge_pkl_file(self, merge_dir=None):
        """
        This function is used to merge the raw output files of PATs codebase.
        The merge_dir contains many pkl files named like: scene_id_range_{begin}_{end}.pkl
        This function also checks continuity
        """
        output_file = "{merge_dir}/matching_infos_{begin}_{end}.pkl"

        def sort_key(file_name):
            numbers = re.findall(r"\d+", file_name)
            return [int(num) for num in numbers]

        # * Load all the files first
        all_files = sorted(os.listdir(merge_dir), key=sort_key)
        merged_info = {}
        first_begin = int(all_files[0].split("_")[3])
        last_end = 0

        for file_name in tqdm(all_files):
            if file_name.endswith(".pkl"):
                # Extract begin and end from file name
                parts = file_name.split("_")
                current_begin = int(parts[3])
                current_end = int(parts[4].split(".")[0])

                # Check for continuity
                if current_begin > last_end:
                    raise ValueError(
                        f"Discontinuity detected: Expected start at {last_end}, but got {current_begin}"
                    )

                # Load and merge the data
                current_path = os.path.join(merge_dir, file_name)
                current_data = mmengine.load(current_path)

                # for all the values in the current_data, if its torch tensor, use cpu().numpy() to change like numpy()
                for scene_id, matchings in current_data.items():
                    for pair_name, matching in matchings.items():
                        if isinstance(matching, dict):
                            current_data[scene_id][pair_name]["kp0"] = (
                                current_data[scene_id][pair_name]["kp0"].cpu().numpy()
                            )
                            current_data[scene_id][pair_name]["kp1"] = (
                                current_data[scene_id][pair_name]["kp1"].cpu().numpy()
                            )

                merged_info.update(current_data)

                # Update the last processed end
                last_end = current_end

                # save the merged_info
                mmengine.dump(
                    merged_info,
                    output_file.format(
                        merge_dir=merge_dir, begin=first_begin, end=last_end
                    ),
                )

        # Store the merged data in the class attribute
        self.infos = merged_info

        # Get the first begin index and the last end index

        print(f"There are in total {len(self.infos)} scenes.")
        return merged_info

    def get_all_matching_pairs(self, scene_id):
        """
        Find all image_ids that have a match with the given image_id within the specified scene_id.
        """

        matching_info = self.infos[scene_id]
        matching_pairs = set(matching_info.keys())
        matching_pairs.discard("processed_num")

        return matching_pairs

    # @TimeCounter(tag="Searching Paired Images", log_interval=20) # costs about 1.6ms
    def get_matching_pairs_from_image_id(self, scene_id, image_id):
        """
        Find all image_ids that have a match with the given image_id within the specified scene_id.
        """
        image_id = self.convert_image_id_to_key(image_id)
        if image_id is None or scene_id not in self.infos:
            print(f"Scene ID {scene_id} not found or image ID {image_id} is invalid.")
            return []

        all_matching_pairs = self.get_all_matching_pairs(scene_id)

        if self.scene_infos is not None:
            # costs 0.3 ms
            all_possible_pairs = self.construct_all_possible_pairs(
                scene_id, image_id
            )  # return is a set
            matching_pairs = all_possible_pairs & all_matching_pairs
        else:
            # check one by one
            # also costs about 0.3 ms
            matching_pairs = set()
            for pair in all_matching_pairs:
                if image_id in pair:
                    matching_pairs.add(pair)  # Add both ids in the pair

        return matching_pairs

    def get_matching_results_by_image_id(self, scene_id, image_id):
        matching_pairs = self.get_matching_pairs_from_image_id(scene_id, image_id)
        # get the results
        matching_results = {pair: self.infos[scene_id][pair] for pair in matching_pairs}
        return matching_results

    def filter_matching_results_by_mask(
        self,
        scene_id,
        image_id,
        mask,
        matching_results,
        remove_matching=True,
        min_matching_num=50,
        cd_loss_thres=0.1,
    ):
        """
        Filter matching keypoints based on a binary mask array.
        Args:
            image_id (str): Image identifier used as a reference in the pair. The mask belongs to this image.
            mask (ndarray): Binary mask array (H x W) where True indicates valid pixels.
            matching_pairs (set): Set of pairs to be processed.
            remove_matching (bool): Wether to remove some mathcing results according to conditions.
            min_matching_num (int): Minimum number of matching keypoints to be kept.
            min_matching_ratio (float): Minimum ratio respective to the valid pixels in the mask of matching keypoints to be kept.
                Note, it's not used.
            camera_verify (bool): Whether to verify the matching using camera projections
        Returns:
            dict: Filtered matching keypoints.
        """
        image_id = self.convert_image_id_to_key(image_id)
        if image_id is None:
            print(f"Image ID {image_id} is invalid.")
            return {}

        # Convert numpy mask to a PyTorch tensor
        mask_tensor = torch.from_numpy(mask).to(self.device)

        filtered_matching_results = {}
        for pair, matching_result in matching_results.items():
            # Determine the order of image_id in the pair
            if image_id == pair[0]:
                source_image_id, target_image_id = pair[0], pair[1]
                source_keypoints, target_keypoints = "kp0", "kp1"
            elif image_id == pair[1]:
                source_image_id, target_image_id = pair[1], pair[0]
                source_keypoints, target_keypoints = "kp1", "kp0"
            else:
                print(f"Image ID {image_id} not found in the pair {pair}.")
                continue

            # Retrieve matching keypoints for the pair
            source_kp = torch.from_numpy(matching_result[source_keypoints]).to(
                self.device
            )
            target_kp = torch.from_numpy(matching_result[target_keypoints]).to(
                self.device
            )

            # Create a mask for valid keypoints
            valid_mask = mask_tensor[source_kp[:, 0].long(), source_kp[:, 1].long()]

            # Filter keypoints based on the mask
            filtered_source_kp = source_kp[valid_mask]
            filtered_target_kp = target_kp[valid_mask]

            # if remove_empty and valid_mask is all False, then skip this pair
            stat_info = {}
            if remove_matching:
                if valid_mask.sum() < min_matching_num:
                    continue

                # verify by projecting the keypoints to the space
                if cd_loss_thres > 0:
                    assert (
                        self.scene_infos is not None
                    ), "When use camery projection to verify matching, should set scene_infos."

                    # generate a mask with keypoints set to True and others set to False
                    filtered_source_mask = torch.zeros_like(mask_tensor)
                    filtered_source_mask[
                        filtered_source_kp.long()[:, 0], filtered_source_kp.long()[:, 1]
                    ] = True

                    filtered_target_mask = torch.zeros_like(mask_tensor)
                    filtered_target_mask[
                        filtered_target_kp.long()[:, 0], filtered_target_kp.long()[:, 1]
                    ] = True

                    # project them to 3d space, takes about 0.2s
                    filtered_source_points_3d = (
                        self.scene_infos.project_image_to_3d_with_mask(
                            scene_id=scene_id,
                            image_id=source_image_id,
                            mask=filtered_source_mask.cpu().numpy(),
                        )
                    )

                    filtered_target_points_3d = (
                        self.scene_infos.project_image_to_3d_with_mask(
                            scene_id=scene_id,
                            image_id=target_image_id,
                            mask=filtered_target_mask.cpu().numpy(),
                        )
                    )

                    # possible that projected points are empty, then skip this pair
                    if (
                        filtered_source_points_3d.shape[0] == 0
                        or filtered_target_points_3d.shape[0] == 0
                    ):
                        continue

                    # calculate the chamfer distance of these two points, using pytorch3d, takes about 0.0s
                    filtered_source_points_3d_tensor = torch.from_numpy(
                        filtered_source_points_3d
                    ).to(self.device)
                    filtered_target_points_3d_tensor = torch.from_numpy(
                        filtered_target_points_3d
                    ).to(self.device)
                    cdloss, _ = chamfer_distance(
                        filtered_source_points_3d_tensor.unsqueeze(0),
                        filtered_target_points_3d_tensor.unsqueeze(0),
                    )

                    # if the cdloss is larger than the threshold, then skip this pair
                    if cdloss > cd_loss_thres:
                        continue

                    stat_info["cdloss"] = cdloss.item()

            # Convert filtered keypoints back to numpy
            filtered_matching_results[pair] = {
                source_keypoints: filtered_source_kp.cpu().numpy(),
                target_keypoints: filtered_target_kp.cpu().numpy(),
                "stat_info": stat_info,
            }

        return filtered_matching_results

    def select_top_k_matching_results_by_cdloss(self, matching_results, top_k):
        """
        This function assumes the `matching_results` has an item 'cdloss' in each matching_result. Sort the matching_results according to cdloss
        and return the top_k items.
        Args:
            matching_results (dict): Dictionary of matching results where each key is a pair of image IDs and each value contains matching keypoints
                                    and additional metadata including 'cdloss'.
            top_k (int): The number of top matching results to return based on the lowest Chamfer distance.
        Returns:
            dict: The top_k matching results sorted by Chamfer distance.
        """
        # Extract the cdloss values and sort them
        sorted_results = sorted(
            matching_results.items(),
            key=lambda x: x[1]["stat_info"]["cdloss"]
            if "stat_info" in x[1] and "cdloss" in x[1]["stat_info"]
            else float("inf"),
        )

        # Select the top_k results based on Chamfer distance
        top_k_results = {pair: result for pair, result in sorted_results[:top_k]}

        return top_k_results

    def visualize_image_matching(
        self, scene_id, matching_result, is_draw_matches=False, save_vis=False
    ):
        """
        Visualize the matching results for a pair of image in a scene. The codes below mainly copies from PATS codes.
        Args:
            scene_id (str): Scene identifier.
            matching_result (dict): Matching pairs and corresponding keypoints between two images. Only one element, {(image_id0, image_id1): {"kp0": kp0s, "kp1": kp1s}}
        Retruns:
            numpy array in BGR format
        """
        image_pair = list(matching_result.keys())[0]
        image_id0, image_id1 = image_pair
        kp0 = matching_result[image_pair]["kp0"].copy()
        kp1 = matching_result[image_pair]["kp1"].copy()

        image_path0 = self.scene_infos.get_image_path(scene_id, image_id0)
        image_path1 = self.scene_infos.get_image_path(scene_id, image_id1)

        # Load images
        image0 = cv2.imread(image_path0)
        image1 = cv2.imread(image_path1)

        output_file_name = f"{scene_id}_{image_id0}_{image_id1}.jpg"

        vis_matching = self.visualizer.vis_matches(
            image0,
            image1,
            kp0,
            kp1,
            is_draw_matches=is_draw_matches,
            save_vis=save_vis,
            file_name=output_file_name,
        )

        return vis_matching

    def save_matching_results_visualization(
        self, scene_id, matching_results, output_dir, is_draw_matches=False
    ):
        mmengine.mkdir_or_exist(output_dir)
        for i, (key, value) in enumerate(matching_results.items()):
            vis_matching = self.visualize_image_matching(
                scene_id, {key: value}, is_draw_matches=is_draw_matches, save_vis=False
            )
            image0, image1 = key
            suffix = ""
            if "stat_info" in value and "cdloss" in value["stat_info"]:
                suffix += f"_cdloss{value['stat_info']['cdloss']:0.3f}"
            cv2.imwrite(
                f"{output_dir}/matching_{i}_{image0}_{image1}{suffix}.jpg", vis_matching
            )


if __name__ == "__main__":
    # Test the class
    scene_infos = SceneInfoHandler(
        "data/scannet/scannet_instance_data/scenes_train_val_info_w_images.pkl"
    )

    handler = MatchingInfoHandler(
        info_path="data/scannet/scannet_match_data/exhaustive_matching.pkl",
        scene_infos=scene_infos,
    )  # for debug
    # handler.check_infos_completeness()

    scene_id = "scene0011_00"
    image_id = "00011"

    # get sam mask
    image_path = scene_infos.get_image_path(scene_id, image_id)
    from ultralytics import SAM

    sam_predictor = SAM("sam_b.pt")
    results = sam_predictor(image_path, bboxes=[[317.76, 272.13, 1296, 917.83]])
    mask = results[0].masks.data.cpu().numpy()[0]
    # save the results
    results[0].save(os.path.basename(image_path).replace(".jpg", "_sam.jpg"))

    matching_results = handler.get_matching_results_by_image_id(scene_id, image_id)
    matching_results = handler.filter_matching_results_by_mask(
        scene_id,
        image_id,
        mask,
        matching_results,
        remove_matching=True,
        min_matching_num=50,
        cd_loss_thres=0.1,
    )
    matching_results = handler.select_top_k_matching_results_by_cdloss(
        matching_results, top_k=5
    )

    for i, (key, value) in enumerate(matching_results.items()):
        vis_matching = handler.visualize_image_matching(
            scene_id, {key: value}, is_draw_matches=True, save_vis=False
        )
        cv2.imwrite(
            f"matching_filtered_5/matching_filtered_{i}_{value['stat_info']['cdloss']}.jpg",
            vis_matching,
        )
