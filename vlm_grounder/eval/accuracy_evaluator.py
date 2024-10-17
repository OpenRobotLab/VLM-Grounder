import argparse
import os
import sys

import cv2
import mmengine
import numpy as np
from tqdm import tqdm

from vlm_grounder.utils import (
    AccuracyCalculator,
    ImageMaskInfoHandler,
    SceneInfoHandler,
    calculate_iou_2d,
    calculate_iou_3d,
)

"""
Usage: 
    python accuracy_evaluator.py --exp_dir {where has the raw result json files} --method {2d_iou|gtbbox_iou|gtbbox_dist}
"""

class AccuracyEvaluator:
    support_methods = ["2d_iou", "gtbbox_iou", "gtbbox_dist"]

    def __init__(
        self,
        exp_dir,
        method="gtbbox_iou",
        info_path="/mnt/afs/xurunsen/projects/vlm_grounder/data/scannet/scannet_instance_data/scenes_train_val_info_w_images.pkl",
    ):
        assert (
            method in self.support_methods
        ), f"Unsupported method: {method}. Supported methods: {self.support_methods}"
        self.exp_dir = exp_dir
        self.info_path = info_path
        self.method = method
        self.accuracy_calculator = AccuracyCalculator()
        self.scene_infos = (
            ImageMaskInfoHandler(info_path)
            if "2d" in method
            else SceneInfoHandler(info_path)
        )
        print(f"AccuracyEvaluator initialized. Method: {method}, exp_dir: {exp_dir}")

    def get_scene_bboxes(self, scene_id: str):
        """
        Returns:
            aligned_bboxes: np.array, shape (num_objects, 6), aligned bounding boxes of all objects in the scene
        """
        num_objects = self.scene_infos.get_num_objects(scene_id)
        aligned_bboxes = []
        for i in range(num_objects):
            aligned_bbox = self.scene_infos.get_object_gt_bbox(
                scene_id=scene_id, object_id=i
            )
            aligned_bboxes.append(aligned_bbox)
        return np.array(aligned_bboxes)

    def get_all_result_json_files(self):
        """
        Find all json files in the experiment directory.
        """
        result_json_files = []
        for file in os.listdir(self.exp_dir):
            if file.endswith(".json"):
                result_json_files.append(file)

        return result_json_files

    def get_raw_result_json_files(self):
        """
        Find all json files in the experiment directory that are not processed yet.
        """
        result_json_files = self.get_all_result_json_files()
        raw_result_json_files = []
        for f in result_json_files:
            flag = 1
            for m in self.support_methods:
                if m in f:
                    flag = 0
                    break
            if flag:
                raw_result_json_files.append(f)
        return raw_result_json_files

    def evaluate_accuracy(self, result_json=None):
        """
        Evaluate the accuracy of the results in the result json file and store the new results in a new json file.
        Args:
            result_json: str or list of str, name or list of result json files, None for auto find all raw result json files.
        """
        if result_json is None:
            result_json_files = self.get_raw_result_json_files()
            # append the self.exp_dir
            result_json_files = [
                os.path.join(self.exp_dir, f) for f in result_json_files
            ]
        elif isinstance(result_json, str):
            result_json_files = [result_json]
            #
            # update the exp_dir, dirname of the result_json
            self.exp_dir = os.path.dirname(result_json)
        else:
            result_json_files = result_json
            self.exp_dir = os.path.dirname(result_json)

        for result_json in result_json_files:
            print(f"Evaluating accuracy from {result_json}")
            result_path = result_json
            new_result_path = result_path.replace(".json", f"_{self.method}.json")

            # check the existence of the result file and the new result file
            if not mmengine.exists(result_path):
                print(
                    f"{result_path} does not exist, which mainly means the from scratch process is not complete."
                )
                continue
            if mmengine.exists(new_result_path):
                print(f"{new_result_path} already exists. Skip.")
                continue

            data = mmengine.load(result_path)
            # print(result_path)
            # print(data)
            results = data["results"]
            print(f"Load {len(results)} results from file {result_path}.")

            # evaluating accuracy
            func = getattr(self, f"evaluate_accuracy_{self.method}")
            new_results = func(results)

            # compute total accuracy and store the results
            accuracy_stat, accuracy_summary = self.accuracy_calculator.compute_accuracy(
                new_results
            )
            final_results = {
                "accuracy_summary": ",".join(accuracy_summary),
                "accuracy_stat": accuracy_stat,
                "total_gpt_cost": data["total_gpt_cost"],
                "results": new_results,
            }
            mmengine.dump(final_results, new_result_path, indent=2)
            print(f"Results saved to {new_result_path}")

        return

    def evaluate_accuracy_2d_iou(self, results):
        """
        Calculate 2D mask IoU for each sample in the results

        """
        for sample in tqdm(results):
            gpt_pred_image_id = sample["gpt_pred_image_id"]
            scene_id = sample["scene_id"]
            query = sample["query"]
            target_id = sample["target_id"]
            pred_target_class = sample["pred_target_class"]
            file_prefix = f"{query.replace('/', 'or').replace(' ', '_')[:60]}"
            intermediate_output_dir = os.path.join(
                self.exp_dir, "intermediate_results", scene_id, file_prefix
            )

            # skip if gpt_pred_image_id is -1
            if gpt_pred_image_id == -1:
                print(
                    f"Warning: gpt_pred_image_id is -1 for scene {scene_id}, target {target_id}. Skip."
                )
                new_eval_result = {"iou2d": -1}
                for key in sample["eval_result"]:
                    if key != "iou3d":
                        new_eval_result[key] = sample["eval_result"][key]
                new_eval_result["acc_iou_25"] = False
                new_eval_result["acc_iou_50"] = False
                sample["eval_result"] = new_eval_result
                continue

            # read predict mask and ground truth mask
            sam_mask_output_dir = os.path.join(
                intermediate_output_dir,
                f"sam_mask_{gpt_pred_image_id:05d}_{pred_target_class.replace('/', 'or')}_target_{target_id}",
            )
            if not os.path.exists(sam_mask_output_dir):
                print(f"Warning: {sam_mask_output_dir} does not exist. Skip.")
                new_eval_result = {"iou2d": -1}
                for key in sample["eval_result"]:
                    if key != "iou3d":
                        new_eval_result[key] = sample["eval_result"][key]
                new_eval_result["acc_iou_25"] = False
                new_eval_result["acc_iou_50"] = False
                sample["eval_result"] = new_eval_result
                continue

            anchor_sam_path = os.path.join(sam_mask_output_dir, "anchor_sam.pkl")
            anchor_sam = mmengine.load(anchor_sam_path)
            pred_mask = anchor_sam.masks.data.numpy()[
                0
            ]  # bool numpy array (width, height)
            gt_mask = self.scene_infos.get_instance_mask(
                scene_id, gpt_pred_image_id, target_id
            )  # bool numpy array (width, height)

            # save the gt_mask
            gt_masked_result = self.scene_infos.apply_transparent_mask(
                scene_id, gpt_pred_image_id, target_id
            )  # return is a ultralytics.engine.results.Results object
            gt_masked_result.save(
                os.path.join(sam_mask_output_dir, "anchor_gt_mask.jpg")
            )

            # calculate iou
            iou = calculate_iou_2d(pred_mask, gt_mask)

            # store iou2d in the sample
            new_eval_result = {"iou2d": iou}
            for key in sample["eval_result"]:
                if key != "iou3d":
                    new_eval_result[key] = sample["eval_result"][key]
            new_eval_result["acc_iou_25"] = iou >= 0.25
            new_eval_result["acc_iou_50"] = iou >= 0.5
            sample["eval_result"] = new_eval_result

        return results

    def evaluate_accuracy_gtbbox_iou(self, results):
        """
        Calculate iou from the predicted bounding box with all instances in the scene
        and select the instance with largest iou as the predicted instance.
        """
        for sample in tqdm(results):
            scene_id = sample["scene_id"]
            target_id = sample["target_id"]
            query = sample["query"]

            # Skip some fail case
            if sample.get("eval_result") is None:
                print(
                    f"Warning: eval_result is None for scene: {scene_id}, target: {target_id}, query: {query}. Skip."
                )
                continue
            if sample["eval_result"].get("gpt_pred_bbox") is None:
                sample["eval_result"]["acc_iou_50"] = False
                sample["eval_result"]["acc_iou_25"] = False
                sample["eval_result"]["iou3d"] = -1
                sample["eval_result"]["max_iou"] = -1
                sample["eval_result"]["min_distance"] = -1
                print(
                    f"Warning: Fail query for scene: {scene_id}, target: {target_id}, query: {query}. Skip."
                )
                continue
            gpt_pred_bbox = np.array(sample["eval_result"]["gpt_pred_bbox"])
            scene_aligned_bboxes = self.get_scene_bboxes(scene_id)

            max_iou = -1
            min_distance = -1

            # calculate iou
            ious = []
            for i in range(scene_aligned_bboxes.shape[0]):
                iou = calculate_iou_3d(gpt_pred_bbox, scene_aligned_bboxes[i])
                ious.append(iou)
            ious = np.array(ious)  # (n, )

            if np.all(ious == 0):
                # if ious are all zero, select use center distance
                print(
                    f"Warning: ious are all zero for scene {scene_id}, target {target_id}. Use center distance."
                )
                centers = scene_aligned_bboxes[:, :3]  # (n, 3)
                center = gpt_pred_bbox[:3]
                distances = np.linalg.norm(centers - center, axis=1)  # (n, )
                selected_idx = np.argmin(distances)
                max_iou = 0.0
                min_distance = distances[selected_idx]
            else:
                selected_idx = np.argmax(ious)
                max_iou = ious[selected_idx]

            if target_id == selected_idx:
                # predict right
                sample["eval_result"]["acc_iou_50"] = True
                sample["eval_result"]["acc_iou_25"] = True
                sample["eval_result"]["iou3d"] = 1
                sample["eval_result"]["max_iou"] = max_iou
                sample["eval_result"]["min_distance"] = min_distance
            else:
                # predict wrong
                sample["eval_result"]["acc_iou_50"] = False
                sample["eval_result"]["acc_iou_25"] = False
                sample["eval_result"]["iou3d"] = 0
                sample["eval_result"]["max_iou"] = max_iou
                sample["eval_result"]["min_distance"] = min_distance

        return results

    def evaluate_accuracy_gtbbox_dist(self, results):
        """
        Select the instance with the smallest center distance as the predicted instance.
        """
        for sample in tqdm(results):
            scene_id = sample["scene_id"]
            target_id = sample["target_id"]
            query = sample["query"]

            # Skip some fail case
            if sample.get("eval_result") is None:
                print(
                    f"Warning: eval_result is None for scene: {scene_id}, target: {target_id}, query: {query}. Skip."
                )
                continue
            if sample["eval_result"].get("gpt_pred_bbox") is None:
                sample["eval_result"]["acc_iou_50"] = False
                sample["eval_result"]["acc_iou_25"] = False
                sample["eval_result"]["iou3d"] = -1
                sample["eval_result"]["max_iou"] = -1
                sample["eval_result"]["min_distance"] = -1
                print(
                    f"Warning: Fail query for scene: {scene_id}, target: {target_id}, query: {query}. Skip."
                )
                continue
            gpt_pred_bbox = np.array(sample["eval_result"]["gpt_pred_bbox"])
            scene_aligned_bboxes = self.get_scene_bboxes(scene_id)

            max_iou = -1
            min_distance = -1

            # calculate center distance
            centers = scene_aligned_bboxes[:, :3]  # (n, 3) xyz
            center = gpt_pred_bbox[:3]
            distances = np.linalg.norm(centers - center, axis=1)  # (n, )
            selected_idx = np.argmin(distances)
            min_distance = distances[selected_idx]

            if target_id == selected_idx:
                # predict right
                sample["eval_result"]["acc_iou_50"] = True
                sample["eval_result"]["acc_iou_25"] = True
                sample["eval_result"]["iou3d"] = 1
                sample["eval_result"]["max_iou"] = max_iou
                sample["eval_result"]["min_distance"] = min_distance
            else:
                # predict wrong
                sample["eval_result"]["acc_iou_50"] = False
                sample["eval_result"]["acc_iou_25"] = False
                sample["eval_result"]["iou3d"] = 0
                sample["eval_result"]["max_iou"] = max_iou
                sample["eval_result"]["min_distance"] = min_distance

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default=None)
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--method", type=str, default=None)
    args = parser.parse_args()

    methods = ["2d_iou", "gtbbox_iou", "gtbbox_dist"]
    if args.method is not None:
        accuracy_evaluator = AccuracyEvaluator(args.exp_dir, method=args.method)
        accuracy_evaluator.evaluate_accuracy(result_json=args.result_path)
    else:
        # all
        for method in methods:
            accuracy_evaluator = AccuracyEvaluator(args.exp_dir, method=method)
            accuracy_evaluator.evaluate_accuracy(
                result_json=args.result_path
            )  # directly setting result_path will change the self.exp_dir

    # print(accuracy_evaluator.get_raw_result_json_files())
