import argparse
import copy
import datetime
import json
import math
import os
import random
import shutil
import sys
import traceback

import cv2
import mmengine
import pandas as pd
import supervision as sv
import torch
from mmengine.utils.dl_utils import TimeCounter
from PIL import Image
from supervision import Position
from supervision.draw.color import Color
from tqdm import tqdm
from ultralytics import SAM
from ultralytics.engine.results import Results

from vlm_grounder.grounder.prompts.visual_grounder_prompts import *
from vlm_grounder.utils import *
from vlm_grounder.utils import (
    DetInfoHandler,
    MatchingInfoHandler,
    OpenAIGPT,
    SceneInfoHandler,
    UltralyticsSAMHuge,
    calculate_iou_3d,
)

DEFAULT_OPENAIGPT_CONFIG = {"temperature": 1, "top_p": 1, "max_tokens": 4095}


class VisualGrounder:
    def __init__(
        self,
        scene_info_path,
        det_info_path,
        vg_file_path,
        output_dir,
        openaigpt_type,
        ensemble_num=1,
        prompt_version=1,
        **kwargs,
    ) -> None:
        self.scene_info_path = scene_info_path
        self.vg_file_path = vg_file_path
        self.output_dir = output_dir
        self.openaigpt_type = openaigpt_type
        self.ensemble_num = ensemble_num

        # cuda available
        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        # set openai
        DEFAULT_OPENAIGPT_CONFIG["temperature"] = kwargs.get("openai_temperature", 1.0)
        DEFAULT_OPENAIGPT_CONFIG["top_p"] = kwargs.get("openai_top_p", 1.0)
        self.openaigpt_config = {"model": openaigpt_type, **DEFAULT_OPENAIGPT_CONFIG}
        self.openaigpt = OpenAIGPT(**self.openaigpt_config)

        # set prompts
        self.prompt_version = prompt_version
        self.system_prompt = SYSTEM_PROMPTS[f"v{prompt_version}"]
        self.input_prompt = INPUT_PROMPTS[f"v{prompt_version}"]
        self.bbox_select_user_prompt = BBOX_SELECT_USER_PROMPTS[f"v{prompt_version}"]
        self.image_id_invalid_prompt = IMAGE_ID_INVALID_PROMPTS[f"v{prompt_version}"]
        self.detection_not_exist_prompt = DETECTION_NOT_EXIST_PROMPTS[
            f"v{prompt_version}"
        ]

        self.evaluate_result_func = eval(
            f'self.{kwargs.get("evaluate_function", "evaluate_3diou")}'
        )

        # set sam
        use_sam_huge = kwargs.get("use_sam_huge", False)
        if use_sam_huge:
            assert os.path.exists(
                "checkpoints/SAM/sam_vit_h_4b8939.pth"
            ), "Error: no checkpoints/SAM/sam_vit_h_4b8939.pth found."
            self.sam_predictor = UltralyticsSAMHuge(
                "checkpoints/SAM/sam_vit_h_4b8939.pth"
            )
            print(f"Use SAM-Huge.")
        else:
            self.sam_predictor = SAM("sam_b.pt")

        self.scene_infos = SceneInfoHandler(scene_info_path)
        self.det_infos = DetInfoHandler(det_info_path)
        self.vg_file = pd.read_csv(vg_file_path)

        self.default_bbox_annotator = sv.BoundingBoxAnnotator()
        self.bbox_annotator = sv.BoundingBoxAnnotator(thickness=6, color=Color.BLACK)
        self.default_label_annotator = sv.LabelAnnotator()
        self.label_annotator = sv.LabelAnnotator(
            text_position=Position.CENTER,
            text_thickness=2,
            text_scale=1,
            color=Color.BLACK,
        )

        self.accuracy_calculator = AccuracyCalculator()

        # matching
        matching_info_path = kwargs.get("matching_info_path", None)
        self.matching_infos = MatchingInfoHandler(
            matching_info_path, scene_infos=self.scene_infos
        )
        self.min_matching_num = kwargs.get("min_matching_num", 50)
        self.cd_loss_thres = kwargs.get("cd_loss_thres", 0.1)
        self.use_point_prompt = kwargs.get("use_point_prompt", False)
        self.use_bbox_prompt = kwargs.get("use_bbox_prompt", False)
        self.use_point_prompt_num = kwargs.get("use_point_prompt_num", 1)
        # assert use_bbox_prompt and use_point_prompt cannot be both False
        assert (
            self.use_bbox_prompt or self.use_point_prompt
        ), "use_bbox_prompt and use_point_prompt cannot be both False."

        # Morphological operations
        self.post_process_erosion = kwargs.get("post_process_erosion", True)
        self.post_process_dilation = kwargs.get("post_process_dilation", True)
        self.kernel_size = kwargs.get("kernel_size", 3)
        self.post_process_component = kwargs.get("post_process_component", True)
        self.post_process_component_num = kwargs.get("post_process_component_num", 1)
        if (
            self.post_process_erosion
            or self.post_process_dilation
            or self.post_process_component
        ):
            self.post_process = True
        else:
            self.post_process = False

        self.point_filter_nb = kwargs.get("point_filter_nb", 20)
        self.point_filter_std = kwargs.get("point_filter_std", 1.0)
        self.point_filter_type = kwargs.get("point_filter_type", "statistical")
        self.point_filter_tx = kwargs.get("point_filter_tx", 0.05)
        self.point_filter_ty = kwargs.get("point_filter_ty", 0.05)
        self.point_filter_tz = kwargs.get("point_filter_tz", 0.05)
        self.project_color_image = kwargs.get("project_color_image", False)

        # set VLM parameters
        self.gpt_max_retry = kwargs.get("gpt_max_retry", 5)
        assert (
            self.gpt_max_retry >= 0
        ), "gpt_max_retry should be greater than or equal to 0."
        self.gpt_max_input_images = kwargs.get("gpt_max_input_images", 6)
        self.get_bbox_index_type = kwargs.get("get_bbox_index_type", "constant")
        self.use_all_images = kwargs.get("use_all_images", False)
        self.use_new_detections = kwargs.get("use_new_detections", False)
        self.skip_bbox_selection_when1 = kwargs.get("skip_bbox_selection_when1", False)
        self.image_det_confidence = kwargs.get("image_det_confidence", 0.3)
        self.use_bbox_anno_f_gpt_select_id = kwargs.get(
            "use_bbox_anno_f_gpt_select_id", False
        )
        self.fix_grid2x4 = kwargs.get("fix_grid2x4", False)
        self.dynamic_stitching = kwargs.get("dynamic_stitching", True)
        self.online_detector = kwargs.get("online_detector", "gdino")

        # set online detector
        if self.use_new_detections:
            if "yolo" in self.online_detector.lower():
                detection_model = "checkpoints/yolov8_world/yolov8x-worldv2.pt"
                self.detection_model = OnlineDetector(
                    detection_model=detection_model, device=self.device
                )
            elif "gdino" in self.online_detector.lower():
                self.detection_model = OnlineDetector(
                    detection_model="Grounding-DINO-1.5Pro", device=self.device
                )

        print(f"Point filter nb: {self.point_filter_nb}, std: {self.point_filter_std}")
        print(
            f"SAM mask processing type, kernel_size: {self.kernel_size}, erosion: {self.post_process_erosion}, dilation: {self.post_process_dilation}, component: {self.post_process_component}, component_num: {self.post_process_component_num}."
        )
        # print(f"Kwargs: {kwargs}.")

        self.output_prefix = f"{os.path.basename(self.vg_file_path.split('.')[0])}_{self.openaigpt_type}_promptv{self.prompt_version}"

    def post_process_mask(self, mask):
        """
        Process a binary mask to smooth and optionally remove small components.

        Args:
            mask (np.array): A 2D numpy array where the mask is boolean (True/False) or is `ultralytics.engine.results.Results`.
            opening (bool): If True, perform morphological opening to remove noise.
            remove_small_component (bool): If True, remove all but the largest connected component.

        Returns:
            np.array: The processed mask as a boolean numpy array.
        """
        is_ultralytics = False
        if isinstance(mask, Results):
            is_ultralytics = True
            ultra_result = mask.new()
            # get the masks
            mask = mask.masks.data.cpu().numpy()[0]

        # Convert boolean mask to uint8
        img = np.uint8(mask) * 255

        # Define the kernel for morphological operations
        kernel = np.ones((self.kernel_size * 2 + 1, self.kernel_size * 2 + 1), np.uint8)

        # Apply morphological erosion if requested
        if self.post_process_erosion:
            img = cv2.erode(img, kernel, iterations=1)

        # Apply morphological dilation if requested
        if self.post_process_dilation:
            img = cv2.dilate(img, kernel, iterations=1)

        # Find all connected components
        num_labels, labels_im = cv2.connectedComponents(
            img
        )  # label 0 is background, so start from 1
        if self.post_process_component and num_labels > 1:
            # Calculate the area of each component and sort them, keeping the largest k
            component_areas = [
                (label, np.sum(labels_im == label)) for label in range(1, num_labels)
            ]
            component_areas.sort(key=lambda x: x[1], reverse=True)
            largest_components = [
                x[0] for x in component_areas[: self.post_process_component_num]
            ]
            img = np.isin(labels_im, largest_components).astype(np.uint8)

        # Return the processed image as a boolean mask
        new_mask = img.astype(bool)

        if is_ultralytics:
            new_mask = torch.from_numpy(new_mask[None, :])  # should be tensor
            ultra_result.update(masks=new_mask)
            return ultra_result

        return new_mask

    def evaluate_3diou(self, pred_bbox, vg_row):
        """
        Evaluate 3D bounding box IoU.
        """
        scene_id = vg_row["scan_id"]
        target_id = vg_row["target_id"]
        gt_bbox = self.scene_infos.get_object_gt_bbox(
            scene_id, object_id=target_id, axis_aligned=True
        )
        # * calculate iou
        iou3d = calculate_iou_3d(pred_bbox, gt_bbox)

        default_eval_result = self.init_default_eval_result_dict(vg_row)

        default_eval_result.update(
            {
                "iou3d": iou3d,
                "acc_iou_25": iou3d >= 0.25,
                "acc_iou_50": iou3d >= 0.50,
                "gpt_pred_bbox": pred_bbox.tolist(),
                "gt_bbox": gt_bbox.tolist(),
            }
        )

        return default_eval_result

    def get_ensemble_masks_matching(
        self, scene_id, image_id, pred_target_class, sam_mask, sam_mask_output_dir
    ):
        """
        Use feature matching to ensemble masks.

        Args:
            scene_id (int): The ID of the scene.
            image_id (int): The ID of the image.
            pred_target_class (str): The predicted target class.
            sam_mask (Tensor): The segmentation mask generated by the SAM model.
            sam_mask_output_dir (str): The output directory to save the generated masks.

        Returns:
            tuple: A tuple containing the ensemble image IDs and ensemble masks.
        """
        if self.ensemble_num == 1:
            return [image_id], [sam_mask]

        matching_results = self.matching_infos.get_matching_results_by_image_id(
            scene_id, image_id
        )
        # filter out those matching pairs with invalid posed images
        matching_results = {
            pair: matching_result
            for pair, matching_result in matching_results.items()
            if self.scene_infos.is_posed_image_valid(scene_id, pair[0])
            and self.scene_infos.is_posed_image_valid(scene_id, pair[1])
        }

        matching_results = self.matching_infos.filter_matching_results_by_mask(
            scene_id,
            image_id,
            sam_mask,
            matching_results,
            remove_matching=True,
            min_matching_num=self.min_matching_num,
            cd_loss_thres=self.cd_loss_thres,
        )

        # by default, the anchor mask is included, so the other images should be self.ensemble_num - 1
        matching_results = self.matching_infos.select_top_k_matching_results_by_cdloss(
            matching_results, top_k=10000
        )  # 10000 here means take all matching results

        # save these matching results
        image_matching_vis_output_dir = os.path.join(
            sam_mask_output_dir, f"ensemble{self.ensemble_num}", "image_matching"
        )
        mmengine.mkdir_or_exist(image_matching_vis_output_dir)

        skipped_image_output_dir = os.path.join(
            sam_mask_output_dir, f"ensemble{self.ensemble_num}", "skipped_images"
        )
        mmengine.mkdir_or_exist(skipped_image_output_dir)

        # For each tracking_images, find the box and mask
        ensemble_image_ids = []
        ensemble_masks = []
        tracking_output_dir = os.path.join(
            sam_mask_output_dir, f"ensemble{self.ensemble_num}", "tracking"
        )
        mmengine.mkdir_or_exist(tracking_output_dir)
        image_id_str = f"{int(image_id):05d}"

        for i, (key, matching_result) in enumerate(matching_results.items()):
            if len(ensemble_masks) == self.ensemble_num - 1:
                # enough masks, break
                break

            # save the matching result visualization
            self.matching_infos.save_matching_results_visualization(
                scene_id,
                {key: matching_result},
                output_dir=image_matching_vis_output_dir,
                is_draw_matches=False,
            )

            if image_id_str == key[0]:
                target_image_id = key[1]
                target_key = "kp1"
            elif image_id_str == key[1]:
                target_image_id = key[0]
                target_key = "kp0"
            else:
                print(
                    f"[Matching Ensemble] Scene id: {scene_id} Image id: {image_id_str} not in matching_pair {key}. Something wrong."
                )
                continue

            target_kps = matching_result[target_key]
            target_image_path = self.scene_infos.get_image_path(
                scene_id, target_image_id
            )

            sam_prompts = {}

            if self.use_bbox_prompt:
                # Get 2d bbox
                ori_detections = self.det_infos.get_detections_filtered_by_score(
                    scene_id, target_image_id, self.image_det_confidence
                )
                detections = self.det_infos.get_detections_filtered_by_class_name(
                    None, None, pred_target_class, ori_detections
                )

                if len(detections) == 0:
                    # no detections
                    all_bboxes_annotated_skipped_image = (
                        self.det_infos.annotate_image_with_detections(
                            scene_id, target_image_id, ori_detections
                        )
                    )
                    all_bboxes_annotated_skipped_image.save(
                        f"{skipped_image_output_dir}/{i}_{target_image_id}_all_bboxes.jpg"
                    )
                    continue
                else:
                    # store the image with target box annotation
                    all_bboxes_annotated_image = (
                        self.det_infos.annotate_image_with_detections(
                            scene_id, target_image_id, detections
                        )
                    )
                    all_bboxes_annotated_image.save(
                        f"{tracking_output_dir}/{i}_{target_image_id}_all_bboxes.jpg"
                    )

                    # first sort the index and then use something like below to get the result
                    # ineed to check if wether iou > 0
                    # use something like detections[0:5][[1, 1, 2, 4]]
                    bbox_areas = detections.box_area  # 1d array

                    # calculate how many keypoints are in the bboxes
                    inliers = [
                        np.sum(
                            (target_kps[:, 1] > x1)
                            & (target_kps[:, 1] < x2)
                            & (target_kps[:, 0] > y1)
                            & (target_kps[:, 0] < y2)
                        )
                        for x1, y1, x2, y2 in detections.xyxy
                    ]

                    iob = inliers / bbox_areas  # intersection over bbox_areas

                    # find the best bbox and nonzero
                    index = np.argmax(iob)

                    if inliers[index] == 0:
                        # print(f"[Matching Ensemble] Scene id: {scene_id}, Target image id: {target_image_id}, Pred target class: {pred_target_class} does not have any keypoints in all bboxes.")
                        continue
                    else:
                        # store the image with target box annotation
                        bbox_annotated_image = (
                            self.det_infos.annotate_image_with_detections(
                                scene_id, target_image_id, detections[[index]]
                            )
                        )
                        bbox_annotated_image.save(
                            f"{tracking_output_dir}/{i}_{target_image_id}_bbox.jpg"
                        )
                        sam_prompts.update(dict(bboxes=detections.xyxy[index]))

            if self.use_point_prompt:
                # selects the self.use_point_prompt_num target_kps with some criteria
                # if use box prompt, then selects the top self.use_point_prompt_num nearest to the bbox center
                # The two versions below are quite similar
                if self.use_bbox_prompt and "bboxes" in sam_prompts:
                    bbox = sam_prompts["bboxes"]
                    bbox_center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                    # calculate the distance
                    distances = np.linalg.norm(
                        target_kps[:, [1, 0]] - bbox_center, axis=1
                    )
                    # get the index
                    selected_kp_index = np.argsort(distances)[
                        : self.use_point_prompt_num
                    ]

                else:
                    # calcualte the center of the kps and select the top_k nearest points
                    kps_center = np.mean(target_kps, axis=0)
                    distances = np.linalg.norm(target_kps - kps_center, axis=1)
                    selected_kp_index = np.argsort(distances)[
                        : self.use_point_prompt_num
                    ]

                selected_target_kps = target_kps[selected_kp_index]
                sam_prompts.update(
                    dict(
                        points=selected_target_kps[:, [1, 0]],
                        labels=np.array([1]).repeat(len(selected_target_kps)),
                    )
                )

                # visualize the keypoints on images
                points_annotated_image = self.det_infos.annotate_image_with_points(
                    scene_id, target_image_id, selected_target_kps[:, [1, 0]]
                )
                # save the image
                points_annotated_image.save(
                    f"{tracking_output_dir}/{i}_{target_image_id}_points.jpg"
                )

            if len(sam_prompts) == 0:
                continue

            prompt_suffix = ""
            if "points" in sam_prompts:
                prompt_suffix += "_w_p"
            if "bboxes" in sam_prompts:
                prompt_suffix += "_w_b"

            # start getting mask
            # Use sam to get the mask
            sam_prediction = self.sam_predictor(
                target_image_path, verbose=False, **sam_prompts
            )[0]

            # save the results
            sam_prediction.save(
                f"{tracking_output_dir}/{i}_{target_image_id}{prompt_suffix}_sam.jpg"
            )

            # postprocessing
            if self.post_process:
                sam_prediction = self.post_process_mask(sam_prediction)
                sam_prediction.save(
                    f"{tracking_output_dir}/{i}_{target_image_id}{prompt_suffix}_sam_postprocessed.jpg"
                )

            ensemble_image_ids.append(target_image_id)
            ensemble_masks.append(sam_prediction.masks.data.cpu().numpy()[0])

        # add the anchor mask and id
        ensemble_image_ids.append(image_id)
        ensemble_masks.append(sam_mask)

        return ensemble_image_ids, ensemble_masks

    def ensemble_pred_points(
        self,
        scene_id,
        image_id,
        pred_target_class,
        sam_mask,
        matched_image_ids,
        sam_mask_output_dir,
        intermediate_output_dir,
        detections,
    ):
        """
        Ensemble the predicted points from different images.

        Args:
            scene_id (str): The scene ID.
            image_id (str or int): The image ID.
            pred_target_class (str): The predicted target class.
            sam_mask (np.ndarray): The predicted segmentation mask.
        Returns:
            ensemble_points (np.ndarray): The ensemble points.

        """

        ensemble_image_ids, ensemble_masks = self.get_ensemble_masks_matching(
            scene_id,
            image_id,
            pred_target_class,
            sam_mask=sam_mask,
            sam_mask_output_dir=sam_mask_output_dir,
        )

        ensemble_points = []
        # projections for all the ids
        for current_image_id, current_mask in zip(ensemble_image_ids, ensemble_masks):
            current_aligned_points_3d = self.scene_infos.project_image_to_3d_with_mask(
                scene_id=scene_id,
                image_id=current_image_id,
                mask=current_mask,
                with_color=self.project_color_image,
            )

            ensemble_points.append(current_aligned_points_3d)

        aligned_points_3d = np.concatenate(ensemble_points, axis=0)

        return aligned_points_3d

    def get_det_bbox3d_from_image_id(
        self,
        scene_id,
        image_id,
        pred_target_class,
        pred_detection,
        gt_object_id,
        query,
        matched_image_ids,  # supposed to have all valid images and not an empty list
        intermediate_output_dir,
        bbox_index=None,
        pred_sam_mask=None,
    ):
        """
        Get the 3D bbox from the image_id.

        Args:
            scene_id (str): The scene ID.
            image_id (str): The image ID
            pred_target_class (str): The predicted target class
            pred_detection (Detection): The predicted detection
            gt_object_id (int): The ground truth object ID
            query (str): The query
            matched_image_ids (list): The matched image IDs
            intermediate_output_dir (str): The intermediate output directory
            bbox_index (int): The bbox index, used while ensembling points
            pred_sam_mask (np.ndarray): The predicted segmentation mask

        Raises:
            NotImplementedError: If the point filter type is not implemented.

        Returns:
            tuple: The predicted 3D bbox.
        """
        image_path = self.scene_infos.get_image_path(scene_id, image_id)
        if image_path is None:
            return None

        image_id = int(image_id)  # str -> int
        sam_mask_output_dir = os.path.join(
            intermediate_output_dir,
            f"sam_mask_{image_id:05d}_{pred_target_class.replace('/', 'or')}_target_{gt_object_id}",
        )

        if pred_sam_mask is not None:
            # pred seg
            result = pred_sam_mask
        else:
            # from scratch
            if pred_detection is None:
                if bbox_index is None:
                    return None
                else:
                    # Get 2d bbox
                    detections = self.det_infos.get_detections_filtered_by_score(
                        scene_id, image_id, self.image_det_confidence
                    )
                    detections = self.det_infos.get_detections_filtered_by_class_name(
                        None, None, pred_target_class, detections
                    )

                    if len(detections) == 0:
                        print(
                            f"[Get Anchor Bbox]The predicted image {image_id} has no detections for {scene_id}: {query}. pred_taget_class: {pred_target_class}."
                        )
                        return None

                    pred_detection = detections[bbox_index]

            bbox_2d = pred_detection.xyxy

            mmengine.mkdir_or_exist(sam_mask_output_dir)

            # Get sam mask from 2d bbox
            # sam_predictor does not support batched inference at the moment, so len(results) is always 1
            # bbox_2d can be n * 4 or 1-d array of 4. If 1-d array, masks.data.shape would still be 1, H, W
            result = self.sam_predictor(image_path, bboxes=bbox_2d, verbose=False)[0]
            # save the mask
            result.save(f"{sam_mask_output_dir}/anchor_sam_raw.jpg")

            mmengine.dump(result, f"{sam_mask_output_dir}/anchor_sam_raw.pkl")
            mmengine.dump(pred_detection, f"{sam_mask_output_dir}/pred_detection.pkl")

            if self.post_process:
                result = self.post_process_mask(result)
                result.save(f"{sam_mask_output_dir}/anchor_sam_postprocessed.jpg")

            mmengine.dump(
                result, f"{sam_mask_output_dir}/anchor_sam_ks{self.kernel_size}.pkl"
            )

            # annotate the bbox on the ori image
            ori_image = Image.open(image_path)
            annotated_image = self.default_bbox_annotator.annotate(
                scene=ori_image, detections=pred_detection
            )
            annotated_image = self.default_label_annotator.annotate(
                scene=annotated_image, detections=pred_detection
            )
            # save the image
            annotated_image.save(f"{sam_mask_output_dir}/anchor_bbox.jpg")

            # save the original image for reference
            shutil.copy(image_path, f"{sam_mask_output_dir}/ancho_ori.jpg")

        # results[0].masks.data.shape would be torch.Size([n, H, W]), n is the number of the input bbox_2d
        sam_mask = result.masks.data.cpu().numpy()[0]
        if pred_detection is None:
            pred_detection = mmengine.load(f"{sam_mask_output_dir}/pred_detection.pkl")
        # ensemble adjacent images
        with TimeCounter(tag="EnsemblePredPoints", log_interval=60):
            aligned_points_3d = self.ensemble_pred_points(
                scene_id,
                image_id,
                pred_target_class,
                sam_mask=sam_mask,
                matched_image_ids=matched_image_ids,
                sam_mask_output_dir=sam_mask_output_dir,
                intermediate_output_dir=intermediate_output_dir,
                detections=pred_detection,
            )

        with TimeCounter(tag="RemoveStatisticalOutliers", log_interval=60):
            if self.point_filter_type == "statistical":
                aligned_points_3d_filtered = remove_statistical_outliers(
                    aligned_points_3d,
                    nb_neighbors=self.point_filter_nb,
                    std_ratio=self.point_filter_std,
                )
            elif self.point_filter_type == "truncated":
                aligned_points_3d_filtered = remove_truncated_outliers(
                    aligned_points_3d,
                    tx=self.point_filter_tx,
                    ty=self.point_filter_ty,
                    tz=self.point_filter_tz,
                )
            elif self.point_filter_type == "none":
                aligned_points_3d_filtered = aligned_points_3d
            else:
                raise NotImplementedError(
                    f"Point filter type {self.point_filter_type} is not implemented."
                )

        # save the aligned points 3d
        aligned_points_3d_output_dir = os.path.join(
            intermediate_output_dir, "projected_points"
        )
        mmengine.mkdir_or_exist(aligned_points_3d_output_dir)
        np.save(
            f"{aligned_points_3d_output_dir}/ensemble{self.ensemble_num}_matching.npy",
            aligned_points_3d,
        )
        np.save(
            f"{aligned_points_3d_output_dir}/ensemble{self.ensemble_num}__matching_filtered.npy",
            aligned_points_3d_filtered,
        )

        # if nan in points, return None
        if (
            aligned_points_3d.shape[0] == 0
            or aligned_points_3d_filtered.shape[0] == 0
            or np.isnan(aligned_points_3d).any()
            or np.isnan(aligned_points_3d_filtered).any()
        ):
            print(
                f"[Box projection] Aligned_points_3d (filtered) is empty or has nan for {scene_id}: {query}. The VLM predicted image id is {image_id}.\
                matched_image_ids: {matched_image_ids}. pred_target_class: {pred_target_class}."
            )
            return None

        pred_bbox = calculate_aabb(aligned_points_3d_filtered)

        return pred_bbox

    def init_default_eval_result_dict(self, vg_row):
        """
        Initializes and returns a default evaluation result dictionary.

        Args:
            vg_row (dict): A dictionary containing the necessary information for evaluation.

        Returns:
            dict: A dictionary with default evaluation results.

        """
        is_unique_scanrefer = vg_row.get("is_unique_scanrefer", False)
        is_multi_scanrefer = not is_unique_scanrefer
        scannet18_class = vg_row.get("scannet18_class", None)

        is_unique_category = vg_row.get("is_unique_category", False)
        is_multi_category = not is_unique_category
        category = vg_row.get("category", None)

        is_easy_referit3d = vg_row.get("is_easy_referit3d", False)
        is_hard_referit3d = not is_easy_referit3d
        distractor_number = vg_row.get("distractor_number", None)

        is_vd_referit3d = vg_row.get("is_vd_referit3d", False)
        is_vid_referit3d = not is_vd_referit3d
        vd_referit3d = vg_row.get("vd_referit3d", None)

        default_eval_result = {
            "iou3d": -1,
            "acc_iou_25": False,
            "acc_iou_50": False,
            "is_unique_scanrefer": is_unique_scanrefer,
            "is_multi_scanrefer": is_multi_scanrefer,
            "scannet18_class": scannet18_class,
            "is_unique_category": is_unique_category,
            "is_multi_category": is_multi_category,
            "category": category,
            "is_easy_referit3d": is_easy_referit3d,
            "is_hard_referit3d": is_hard_referit3d,
            "distractor_number": distractor_number,
            "is_vd_referit3d": is_vd_referit3d,
            "is_vid_referit3d": is_vid_referit3d,
            "vd_referit3d": vd_referit3d,
            "gpt_pred_bbox": None,
        }

        return default_eval_result

    def get_bbox_index_constant(self):
        return 0

    def get_gpt_select_bbox_index(
        self, messages, scene_id, image_id, query, detections, intermediate_output_dir
    ):
        """
        Use VLM to choose which bounding box is the target object in one image.

        Parameters:
            messages (list): A list of messages exchanged between the user and the assistant.
            scene_id (int): The ID of the scene.
            image_id (int): The ID of the image.
            query (str): The query string.
            detections (list): A list of bounding box detections.
            intermediate_output_dir (str): The directory to store intermediate output.

        Returns:
            tuple: A tuple containing the following elements:
                - bbox_index (int): The index of the selected bounding box.
                - message_his (list): The history of messages exchanged between the user and the assistant.
                - cost (int): The cost of the VLM operation.
                - retry (int): The number of retries performed.

        Raises:
            NotImplementedError: If the prompt version is greater than 3.

        """
        num_candidate_bboxes = len(detections)
        if self.prompt_version <= 2:
            bbox_select_user_prompt = self.bbox_select_user_prompt.format(
                num_candidate_bboxes=num_candidate_bboxes,
                query=query,
                image_id=image_id,
            )
        elif self.prompt_version <= 3:
            bbox_select_user_prompt = self.bbox_select_user_prompt.format(
                num_candidate_bboxes=num_candidate_bboxes
            )
        else:
            raise NotImplementedError

        # Annotate the image with detectionss
        image_path = self.scene_infos.get_image_path(scene_id, image_id)
        assert image_path, "\t[GPTSelectBBoxID] The image path is None when letting VLM to select bbounding index. scene_id: {scene_id}, image_id: {image_id}."

        bbox_index_gpt_select_output_dir = os.path.join(
            intermediate_output_dir, "bbox_index_gpt_select_prompts"
        )
        mmengine.mkdir_or_exist(bbox_index_gpt_select_output_dir)

        image = Image.open(image_path)

        labels = [f"ID:{ID}" for ID in range(len(detections))]
        if self.use_bbox_anno_f_gpt_select_id:
            image = self.bbox_annotator.annotate(scene=image, detections=detections)
        annotated_image = self.label_annotator.annotate(
            scene=image, detections=detections, labels=labels
        )
        # save the annotated_image
        annotated_image.save(
            f"{bbox_index_gpt_select_output_dir}/{image_id:05d}_annotated.jpg"
        )

        # convert the annotated image to base64
        annotated_image_base64 = encode_PIL_image_to_base64(
            resize_image(annotated_image)
        )

        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": bbox_select_user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{annotated_image_base64}",
                            "detail": "high",
                        },
                    },
                ],
            }
        )

        # call VLM to get the result
        cost = 0
        retry = 1
        while retry <= self.gpt_max_retry:
            bbox_index = -1
            gpt_message = None
            try:
                mmengine.dump(
                    filter_image_url(messages),
                    f"{bbox_index_gpt_select_output_dir}/{image_id:05d}_gpt_prompts_retry{retry}.json",
                    indent=2,
                )
                gpt_response = self.openaigpt.safe_chat_complete(
                    messages, response_format={"type": "json_object"}, content_only=True
                )
            except Exception as e:
                print(
                    f"\t[GPTSelectBBoxID] {scene_id}: [{query[:20]}] VLM failed to return a response. Error: {e} Retrying..."
                )
                retry += 1
                continue

            cost += gpt_response["cost"]
            gpt_content_json = json.loads(gpt_response["content"])
            gpt_message = {
                "role": "assistant",
                "content": [{"text": gpt_response["content"], "type": "text"}],
            }
            messages.append(gpt_message)
            # save gpt_content_json
            mmengine.dump(
                filter_image_url(messages),
                f"{bbox_index_gpt_select_output_dir}/{image_id:05d}_gpt_responses_retry{retry}.json",
                indent=2,
            )

            bbox_index = int(gpt_content_json.get("object_id", -1))
            if bbox_index < 0 or bbox_index >= num_candidate_bboxes:
                # invalid bbox index
                retry += 1
                bbox_invalid_prompt = f"""Your selected bounding box ID: {bbox_index} is invalid, it should start from 0 and there are only {num_candidate_bboxes} candidate objects in the image. Now try again to select one. Remember to reply using JSON format with the required keys."""

                print(
                    f"\t[VLMSelectBBoxID] {scene_id}: [{query[:20]}] VLM gives an invalid bbox ID {bbox_index}. Append the response and retrying..."
                )
                messages.append({"role": "user", "content": bbox_invalid_prompt})
                print(
                    f"\t[VLMSelectBBoxID] {scene_id}: [{query[:20]}] Here are the messages."
                )
                continue
            else:
                break

        if isinstance(messages[-1], dict) and messages[-1]["role"] == "user":
            # means the process is not complete, whether VLM produces no response or VLM provides invalid bbox id
            # the outer module should start from the begining
            message_his = None
        else:
            # means, VLM give a valid bbox_index
            message_his = messages
        return bbox_index, message_his, cost, retry

    def get_gpt_pred_image_id(self, scene_id, query, messages, intermediate_output_dir):
        """
        Use VLM to select the image containing the target object.

        Args:
            scene_id (str): The ID of the scene.
            query (str): The query string.
            messages (list): A list of messages exchanged between the user and the assistant.
            intermediate_output_dir (str): The directory to store intermediate output.

        Returns:
            tuple: A tuple containing the following elements:
                - pred_image_id (int): The predicted image ID.
                - gpt_content_json (dict): The JSON content of the GPT response.
                - message_his (list): The updated list of messages exchanged between the user and the assistant.
                - cost (int): The total cost of the GPT responses.
                - retry (int): The number of retries made.

        Raises:
            Exception: If VLM fails to return a response.

        """
        image_id_gpt_select_output_dir = os.path.join(
            intermediate_output_dir, "image_index_gpt_select_prompts"
        )
        mmengine.mkdir_or_exist(image_id_gpt_select_output_dir)

        cost = 0
        retry = 1
        pred_image_id = -1
        gpt_content_json = ""
        while retry <= self.gpt_max_retry:
            ###### * Get pred_image_id use VLM
            try:
                # save the prompt
                mmengine.dump(
                    filter_image_url(messages),
                    f"{image_id_gpt_select_output_dir}/gpt_prompts_retry{retry}.json",
                    indent=2,
                )
                # call VLM api
                with TimeCounter(tag="pred_image_id VLM Time", log_interval=10):
                    gpt_response = self.openaigpt.safe_chat_complete(
                        messages,
                        response_format={"type": "json_object"},
                        content_only=True,
                    )
            except Exception as e:
                print(
                    f"[VLMPredImageID] {scene_id}: [{query[:20]}] VLM failed to return a response. Error: {e} Retrying..."
                )
                retry += 1
                continue

            # Extract and process the relevant information from the response
            cost += gpt_response["cost"]
            gpt_content = gpt_response["content"]
            gpt_content_json = json.loads(gpt_content)
            pred_image_id = gpt_content_json.get("target_image_id", -1)

            # save the response
            gpt_message = {
                "role": "assistant",
                "content": [{"text": gpt_content, "type": "text"}],
            }
            mmengine.dump(
                filter_image_url(messages + [gpt_message]),
                f"{image_id_gpt_select_output_dir}/gpt_responses_retry{retry}.json",
                indent=2,
            )

            image_path = self.scene_infos.get_image_path(scene_id, pred_image_id)
            if image_path is None:
                # image ID invalid
                retry += 1
                print(
                    f"[VLMPredImageID] {scene_id}: [{query[:20]}] VLM failed to find a valid image ID. Pred image id is {pred_image_id}. Retrying..."
                )

                # append the gpt response
                if self.image_id_invalid_prompt is not None:
                    messages.append(gpt_message)
                    image_id_invalid_prompt = self.image_id_invalid_prompt.format(
                        image_id=pred_image_id
                    )
                    messages.append(
                        {"role": "user", "content": image_id_invalid_prompt}
                    )
                continue

            messages.append(gpt_message)
            pred_image_id = int(pred_image_id)  # str -> int, get a valid image ID
            break
            ###### * End of get pred_image_id using VLM

        if isinstance(messages[-1], dict) and messages[-1]["role"] == "user":
            # means the process is not complete, whether VLM produces no response or VLM provides invalid image id
            message_his = None
        else:
            # means VLM gives a valid image id
            message_his = messages
        return pred_image_id, gpt_content_json, message_his, cost, retry

    def get_image_id_bbox_vanilla(self, vg_row):
        """
        Main process, use VLM to get the target object's bounding box.

        Args:
            vg_row (dict): A dictionary containing the information related to the visual grounding task.

        Returns:
            dict: A dictionary containing the following information:
                - done (bool): Indicates whether the VLM process was successful or not.
                - pred_image_id (int): The predicted image ID.
                - bbox_index (int): The index of the predicted bounding box.
                - gpt_input (str): The input prompt for the VLM model.
                - gpt_content (str): The JSON string containing the content generated by the VLM model.
                - gpt_cost (float): The total cost of the VLM process.
                - pred_detection (dict): The predicted detection information for the target object.
                - other_infos (dict): Additional information about the VLM process, including the number of iterations in the main loop, the number of times the predicted image ID was obtained, and more.
        """
        # Extract scan_id and query from the vg_row
        scene_id = vg_row["scan_id"]
        query = vg_row["utterance"]
        pred_target_class = vg_row["pred_target_class"]
        conditions = eval(vg_row["attributes"]) + eval(vg_row["conditions"])
        matched_image_ids = vg_row["matched_image_ids"]
        if self.use_all_images:
            # get all image ids
            num_posed_images = self.scene_infos.get_num_posed_images(scene_id)
            matched_image_ids = [f"{i:05d}" for i in range(num_posed_images)]
            # filter out those with invalid extrinsics, not possible that all match_image_ids are invalid
            matched_image_ids = [
                match_image_id
                for match_image_id in matched_image_ids
                if self.scene_infos.is_posed_image_valid(scene_id, match_image_id)
            ]

        intermediate_output_dir = vg_row["intermediate_output_dir"]

        ###### * Load images and create grid images
        views_pre_selection_paths = [
            self.det_infos.get_image_path(scene_id, image_id)
            for image_id in matched_image_ids
        ]
        images = [Image.open(img_path) for img_path in views_pre_selection_paths]

        base64Frames = self.stitch_and_encode_images_to_base64(
            images, matched_image_ids, intermediate_output_dir=intermediate_output_dir
        )
        ###### * End of loading images and creating grid images

        ###### * Format the prompt for VLM
        input_prompt_dict = {
            "query": query,
            "pred_target_class": pred_target_class,
            "conditions": conditions,
            "num_view_selections": len(matched_image_ids),
        }

        if self.prompt_version <= 2:
            gpt_input = self.input_prompt.format_map(input_prompt_dict)
            system_prompt = ""
        elif self.prompt_version <= 3:  # support
            gpt_input = self.input_prompt.format_map(input_prompt_dict)
            system_prompt = self.system_prompt
        else:
            raise NotImplementedError

        begin_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": gpt_input},
                    *map(
                        lambda x: {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{x}",
                                "detail": "high",
                            },
                        },
                        base64Frames,
                    ),
                ],
            },
        ]

        image_id_gpt_select_output_dir = os.path.join(
            intermediate_output_dir, "image_index_gpt_select_prompts"
        )
        mmengine.mkdir_or_exist(image_id_gpt_select_output_dir)
        ###### * End of format the prompt for VLM

        retry = 1
        pred_image_id_retry = 0
        select_bbox_index_retry = 0
        messages = begin_messages
        total_cost = 0
        pred_detection = None
        bbox_index = None
        is_gpt_select_bbox_index = False
        gpt_content_json = ""

        # * main loop
        while retry <= self.gpt_max_retry:
            ###### * Get pred_image_id using VLM
            pred_image_id, gpt_content_json, message_his, cost, retry_times = (
                self.get_gpt_pred_image_id(
                    scene_id, query, messages, intermediate_output_dir
                )
            )
            total_cost += cost
            pred_image_id_retry += retry_times
            if message_his is not None:
                # successfully get a valid image id
                image_path = self.scene_infos.get_image_path(scene_id, pred_image_id)
                messages = message_his  # include stitched images
            elif message_his is None:
                # not complete, retry
                retry += 1
                print(
                    f"[GetImageIDBboxVanilla] {scene_id}: [{query[:20]}] VLM predict image id module is not complete. This may because VLM doet not respond or the responded image ID is invalid. Retrying from getting a image id."
                )
                continue
            ###### * End of get pred_image_id using VLM

            ###### * Get pred_bbox_id
            if self.use_new_detections:
                # use online detector
                new_detection_output_dir = os.path.join(
                    intermediate_output_dir, "re_detection_results"
                )
                mmengine.mkdir_or_exist(new_detection_output_dir)
                detections = self.detection_model.detect(image_path, pred_target_class)
                # save the online detection results
                det_image = Image.open(image_path)
                annotated_image = self.default_bbox_annotator.annotate(
                    scene=det_image, detections=detections
                )
                # save the annotated_image
                annotated_image.save(
                    f"{new_detection_output_dir}/{pred_image_id:05d}_new_det_results_retry{retry}.jpg"
                )
            else:
                # get detection from database
                detections = self.det_infos.get_detections_filtered_by_score(
                    scene_id, pred_image_id, self.image_det_confidence
                )
                detections = self.det_infos.get_detections_filtered_by_class_name(
                    None, None, pred_target_class, detections
                )

            if len(detections) == 0:
                # can not detect any target object
                retry += 1
                print(
                    f"[GetImageIDBboxVanilla] {scene_id}: [{query[:20]}] The predicted image {pred_image_id} has no detections for pred_target_class {pred_target_class}. Retrying..."
                )
                if self.detection_not_exist_prompt is not None:
                    detection_not_exist_prompt = self.detection_not_exist_prompt.format(
                        image_id=pred_image_id, pred_target_class=pred_target_class
                    )
                    messages.append(
                        {"role": "user", "content": detection_not_exist_prompt}
                    )
                continue
            elif len(detections) == 1 and self.skip_bbox_selection_when1:
                # only one target object
                bbox_index = 0
            else:
                # more than one target object
                if self.get_bbox_index_type == "constant":
                    # use a constant bbox id
                    bbox_index = self.get_bbox_index_constant()
                elif self.get_bbox_index_type == "vlm_select":
                    ###### * Get pred_bbox_id using VLM
                    print("[GetImageIDBboxVanilla] Use VLM to select the bounding box index.")
                    new_messages = copy.deepcopy(messages)
                    bbox_index, message_his, cost, retry_times = (
                        self.get_gpt_select_bbox_index(
                            new_messages,
                            scene_id=scene_id,
                            image_id=pred_image_id,
                            query=query,
                            detections=detections,
                            intermediate_output_dir=intermediate_output_dir,
                        )
                    )

                    total_cost += cost
                    select_bbox_index_retry += retry_times

                    if message_his is not None:
                        # successfully get a bbox_index
                        is_gpt_select_bbox_index = True
                    elif message_his is None:
                        # Means the interaction in the gpt_selecting id module is discarded
                        retry += 1
                        print(
                            f"[GetImageIDBboxVanilla] {scene_id}: [{query[:20]}] VLM selecting bbox index module is not complete. This may because VLM doet not respond or the responded BBox ID is invalid. Retrying from getting a image id."
                        )
                        messages = begin_messages  # reset messages
                        continue
                    ###### * End get pred_bbox_id using VLM
            ###### * End get pred_bbox_id

            pred_detection = detections[bbox_index]
            break

        # judge if success or not for the VLM process
        if retry > self.gpt_max_retry:
            # VLM Fail in the above process
            print(
                f"[GetImageIDBboxVanilla] {scene_id}: [{query[:20]}] VLM failed to find a valid image or a valid bbox after {self.gpt_max_retry} retries."
            )
            # random choose one
            pred_detection = None
            bbox_index = None
            pred_image_id = -1

        data_dict = {
            "done": retry <= self.gpt_max_retry,
            "pred_image_id": pred_image_id,
            "bbox_index": bbox_index,
            "gpt_input": gpt_input,
            "gpt_content": gpt_content_json,
            "gpt_cost": total_cost,
            "pred_detection": pred_detection,
            "other_infos": {
                "main_loop_times": retry,
                "pred_image_id_times": pred_image_id_retry,
                "select_bbox_index_times": (
                    select_bbox_index_retry if is_gpt_select_bbox_index else None
                ),
                "is_gpt_select_bbox_index": is_gpt_select_bbox_index,
            },
        }
        return data_dict

    @TimeCounter(tag="stitch_and_encode_images_to_base64 Time", log_interval=50)
    def stitch_and_encode_images_to_base64(
        self, images, images_id, intermediate_output_dir=None
    ):
        """
        Stitch and encode a list of images into a grid and convert them to base64 format.

        Args:
            images (list): A list of PIL images to be stitched.
            images_id (list): A list of image IDs corresponding to the images.
            intermediate_output_dir (str, optional): The directory to save the grid images. Defaults to None.

        Returns:
            list: A list of base64 encoded images.

        """
        # determine layouts
        if self.fix_grid2x4:
            # fix layout
            row = 2
            column = 4
        else:
            # square stitching
            num_views_pre_selection = len(images)
            grid_nums = math.ceil(num_views_pre_selection / self.gpt_max_input_images)
            column = math.ceil(grid_nums**0.5)
            row = math.ceil(grid_nums / column)
        grid_layout = (row, column)

        # stitch images
        if self.dynamic_stitching:
            grid_images = dynamic_stitch_images_fix_v2(
                images,
                fix_num=self.gpt_max_input_images,
                ID_array=images_id,
                ID_color="red",
                annotate_id=True,
            )
        else:
            grid_images = stitch_images(
                images,
                grid_dims=grid_layout,
                ID_array=images_id,
                ID_color="red",
                annotate_id=True,
            )

        # save all the grid images
        if intermediate_output_dir is not None:
            grid_images_output_dir = os.path.join(
                intermediate_output_dir, "grid_images"
            )
            mmengine.mkdir_or_exist(grid_images_output_dir)

            # save one by one
            for index, grid_image in enumerate(grid_images):
                grid_image.save(f"{grid_images_output_dir}/grid{index}.jpg")

        # encode to base64
        base64Frames = [
            encode_PIL_image_to_base64(resize_image(image)) for image in grid_images
        ]
        return base64Frames

    def eval_worker_image_only(self, vg_row):
        """
        Evaluate the worker for image-only grounding.

        Args:
            vg_row (pandas.Series): The row containing the information for evaluation.

        Returns:
            dict: A dictionary containing the evaluation results.

        """
        scene_id = vg_row["scan_id"]
        query = vg_row["utterance"]
        target_id = vg_row["target_id"]
        pred_target_class = vg_row["pred_target_class"]
        matched_image_ids = eval(
            vg_row[f"matched_image_ids_confidence{self.image_det_confidence}"]
        )

        if len(matched_image_ids) == 0:
            # Use all images
            num_posed_images = self.scene_infos.get_num_posed_images(scene_id)
            print(
                f"[PrepareInitialInput] No matched images for {scene_id}: {query}. Using all {num_posed_images} images."
            )
            matched_image_ids = [f"{i:05d}" for i in range(num_posed_images)]

        # filter out those with invalid extrinsics, not possible that all match_image_ids are invalid
        matched_image_ids = [
            match_image_id
            for match_image_id in matched_image_ids
            if self.scene_infos.is_posed_image_valid(scene_id, match_image_id)
        ]
        assert (
            len(matched_image_ids) > 0
        ), f"All matched_image_ids are invalid in {scene_id}. This is very strange!"

        vg_row["matched_image_ids"] = matched_image_ids
        file_prefix = f"{query.replace('/', 'or').replace(' ', '_')[:60]}"
        intermediate_output_dir = os.path.join(
            self.output_dir, "intermediate_results", scene_id, file_prefix
        )
        vg_row["intermediate_output_dir"] = intermediate_output_dir

        default_result_dict = {
            "done": False,
            "scene_id": scene_id,
            "query": query,
            "pred_target_class": pred_target_class,
            "gpt_input": None,
            "gpt_content": None,
            "gpt_pred_image_id": None,
            "target_id": target_id,
            "gpt_cost": 0,
            "other_infos": None,
            "eval_result": self.init_default_eval_result_dict(vg_row),
        }

        # The input dict for the grounding modules
        data_dict = self.get_image_id_bbox_vanilla(vg_row)

        # Extract the relevant information from the data_dict
        pred_image_id = data_dict["pred_image_id"]
        bbox_index = data_dict["bbox_index"]
        gpt_input = data_dict["gpt_input"]
        gpt_content = data_dict["gpt_content"]
        gpt_cost = data_dict["gpt_cost"]
        pred_detection = data_dict["pred_detection"]
        other_infos = data_dict["other_infos"]
        done = data_dict["done"]

        default_result_dict.update(
            {
                "done": done,
                "gpt_input": gpt_input,
                "gpt_content": gpt_content,
                "gpt_pred_image_id": pred_image_id,
                "bbox_index": bbox_index,
                "gpt_cost": gpt_cost,
                "other_infos": other_infos,
            }
        )

        # * get the pred 3d bouding box from pred_image_id
        pred_bbox = self.get_det_bbox3d_from_image_id(
            scene_id,
            pred_image_id,
            pred_target_class,
            pred_detection=pred_detection,
            gt_object_id=target_id,
            query=query,
            matched_image_ids=matched_image_ids,
            intermediate_output_dir=intermediate_output_dir,
        )

        if pred_bbox is None:
            return default_result_dict

        eval_result = self.evaluate_result_func(pred_bbox, vg_row)

        default_result_dict.update({"eval_result": eval_result})

        return default_result_dict

    def save_progress(self, results, output_file):
        """
        Save temp result to file.
        """
        mmengine.mkdir_or_exist(self.output_dir)
        output_path = os.path.join(self.output_dir, output_file)
        mmengine.dump(results, output_path, indent=2)

        print(f"Results saved to {output_path}")

    def load_progress(self, temp_output_file):
        """
        Load temp result from file.
        """
        temp_output_path = os.path.join(self.output_dir, temp_output_file)
        if os.path.exists(temp_output_path):
            temp_results = mmengine.load(temp_output_path)
            if "results" in temp_results:
                return temp_results["results"]
            else:
                return temp_results
        return None

    def remove_temp(self, temp_output_file):
        """
        Remove temp result file.
        """
        temp_output_path = os.path.join(self.output_dir, temp_output_file)
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
            print(f"Removed {temp_output_path}")

    def eval(self):
        """
        Evaluate the visual grounder.

        This method performs the evaluation of the visual grounder by processing tasks and generating results.
        It checks if the results already exist and returns if they do.
        If there are resumed results from a previous evaluation, it resumes from there.
        Otherwise, it starts a new evaluation.
        The evaluation is performed in parallel for multiple tasks using tqdm for progress tracking.
        The progress is periodically saved to a temporary file.
        After the evaluation is complete, the final results are calculated and saved to the output file.
        If the evaluation is interrupted, the progress is saved to the temporary file.

        Returns:
            final_results (list): The final results of the evaluation.
        """
        print(f"Start evaluating {self.vg_file_path}.")
        temp_output_file = f"{self.output_prefix}_temp_results.json"
        output_file = f"{self.output_prefix}_results.json"
        # * if output_file already exists, means done
        if mmengine.exists(os.path.join(self.output_dir, output_file)):
            print(f"Results already exist in {output_file}.")
            return
        resumed_results = self.load_progress(temp_output_file)
        self.eval_worker = self.eval_worker_image_only

        if resumed_results:
            # assert resumed_results should be a list
            assert isinstance(
                resumed_results, list
            ), "resumed_results should be a list."
            resumed_results = [
                result for result in resumed_results if result["done"]
            ]  # only include done results
            print(
                f"Resuming from saved progress in {temp_output_file}. Already processed {len(resumed_results)} results."
            )
        else:
            print("Starting new evaluation")
            resumed_results = []

        completed_pairs = set(
            (result["scene_id"], result["query"]) for result in resumed_results
        )

        # Tasks that need to be processed are those not in completed_pairs
        tasks = [
            row
            for _, row in self.vg_file.iterrows()
            if (row["scan_id"], row["utterance"]) not in completed_pairs
        ]
        print(f"{len(tasks)} tasks remaining to be processed.")

        random.shuffle(tasks)

        try:  # * we want to save temp results, so cannot use mmengine.track_parallel_progress
            with tqdm(total=len(tasks)) as pbar:
                for task in tasks:
                    result = self.eval_worker(task)
                    resumed_results.append(result)
                    pbar.update()

                    # Periodically save progress
                    if pbar.n % 50 == 0:
                        self.save_progress(resumed_results, temp_output_file)

            final_results, is_complete = self.calculate_final_results(resumed_results)
            if is_complete:
                self.save_progress(final_results, output_file)
                # Remove the temporary file after successful completion
                self.remove_temp(temp_output_file)
            else:
                self.save_progress(final_results, temp_output_file)

            return final_results

        except (Exception, KeyboardInterrupt) as e:
            print(
                f"Error of type: {type(e).__name__}, message: {e} occurred during evaluation."
            )
            traceback.print_exc()
            final_results, _ = self.calculate_final_results(resumed_results)
            self.save_progress(final_results, temp_output_file)
            print(f"Progress is saved.")

    def calculate_final_results(self, results):
        """
        Calculate final accuracy.
        """
        results, is_complete = self.validate_final_results(results)

        accuracy_stat, accuracy_summary = self.accuracy_calculator.compute_accuracy(
            results
        )

        self.accuracy_calculator.print_statistics()

        total_cost = self.get_costs(results)

        return {
            "accuracy_summary": ",".join(accuracy_summary),
            "accuracy_stat": accuracy_stat,
            "total_gpt_cost": total_cost,
            "results": results,
        }, is_complete

    def validate_final_results(self, results):
        """
        Verify results, check if all task is complete, also remove extra results.
        """
        print("Verifying results...")
        # Get all unique scan_id and query pairs from the vg file
        vg_pairs = set(
            self.vg_file[["scan_id", "utterance"]].itertuples(index=False, name=None)
        )

        # Get all unique scan_id and query pairs from the results
        result_pairs = set((result["scene_id"], result["query"]) for result in results)

        is_complete = True

        # Check if all pairs in the vg file are in the results
        if vg_pairs != result_pairs:
            missing_pairs = vg_pairs - result_pairs
            extra_pairs = result_pairs - vg_pairs

            # Report missing and extra pairs
            if missing_pairs:
                print(
                    f"Missing pairs: {missing_pairs}. {len(missing_pairs)} pairs are missed."
                )
                is_complete = False
            if extra_pairs:
                print(
                    f"Extra pairs: {extra_pairs}. {len(extra_pairs)} extra pairs will be removed."
                )

            # Remove extra pairs from results
            filtered_results = [
                result
                for result in results
                if (result["scene_id"], result["query"]) not in extra_pairs
            ]

            # Optionally, update results list
            results[:] = filtered_results

            print(
                f"Results have been updated to remove extra pairs. Now it has {len(results)} results."
            )
        else:
            print(
                "Verified Success. All scan_id and query pairs in the vg file were matched."
            )

        # sort the resuts according to "scene_id"
        results = sorted(results, key=lambda x: x["scene_id"])

        return results, is_complete

    def get_costs(self, results):
        """
        Get total VLM cost.
        """
        cost = sum(result["gpt_cost"] for result in results)
        return cost


def get_csv_row(vg_file, scene_id, query):
    matching_row = vg_file[
        (vg_file["scan_id"] == scene_id) & (vg_file["utterance"] == query)
    ]

    if not matching_row.empty:
        return matching_row.iloc[0]
    else:
        return "No matching row found."


class Tee:
    """
    Show log in terminal and log file.
    """

    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="outputs/visual_grounding"
    )
    parser.add_argument(
        "--scene_info_path",
        type=str,
        default="data/scannet/scannet_instance_data/scenes_train_val_info_w_images.pkl",
    )  # * if change this, need to disable global cache of tools
    parser.add_argument(
        "--det_info_path",
        type=str,
        default="outputs/image_instance_detector/Grounding-DINO-1_scanrefer_test_top250_pred_target_classes/chunk-1/detection.pkl",
    )
    parser.add_argument(
        "--vg_file_path",
        type=str,
        default="outputs/query_analysis/scanrefer_250.csv",
    )
    parser.add_argument(
        "--matching_info_path",
        type=str,
        default="data/scannet/scannet_match_data/exhaustive_matching.pkl",
    )

    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--result_path", type=str, default=None)

    # sam setting
    parser.add_argument("--use_sam_huge", action="store_true")
    parser.add_argument(
        "--evaluate_function",
        type=str,
        default="evaluate_3diou",
        choices=["evaluate_3diou"],
    )
    # openai setting
    parser.add_argument("--openaigpt_type", type=str, default="gpt-4o-2024-05-13")
    parser.add_argument("--gpt_max_retry", type=int, default=3)
    parser.add_argument("--gpt_max_input_images", type=int, default=6)
    parser.add_argument("--use_all_images", action="store_true", help="Use all images for grounding without view_preselection.")
    parser.add_argument("--ensemble_num", type=int, default=7)  # 1 means no ensemble
    parser.add_argument("--openai_top_p", type=float, default=0.3)
    parser.add_argument("--openai_temperature", type=float, default=0.1)

    # post_processing
    parser.add_argument("--use_bbox_prompt", action="store_true")
    parser.add_argument("--use_point_prompt", action="store_true")
    parser.add_argument("--use_point_prompt_num", type=int, default=1)
    parser.add_argument("--post_process_erosion", action="store_true")
    parser.add_argument("--post_process_dilation", action="store_true")
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument("--post_process_component", action="store_true")
    parser.add_argument("--post_process_component_num", type=int, default=2)
    parser.add_argument("--point_filter_nb", type=int, default=5)
    parser.add_argument("--point_filter_std", type=float, default=1.0)
    parser.add_argument(
        "--point_filter_type",
        type=str,
        default="statistical",
        choices=["statistical", "truncated", "none"],
    )
    parser.add_argument("--point_filter_tx", type=float, default=0.05)
    parser.add_argument("--point_filter_ty", type=float, default=0.05)
    parser.add_argument("--point_filter_tz", type=float, default=0.05)
    parser.add_argument("--project_color_image", default=True, action="store_true")
    parser.add_argument("--use_bbox_anno_f_gpt_select_id", action="store_true")

    # For ensemble experiments
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--search_ensemble", action="store_true", help="Directly use the grounding intermediate outputs and perform more ensemble projection with different ensemble number.")

    # exp name
    parser.add_argument("--exp_name", default=None, type=str)

    # For get bbox index type
    parser.add_argument("--skip_bbox_selection_when1", action="store_true")
    parser.add_argument("--prompt_version", type=int, default=3) 
    parser.add_argument("--image_det_confidence", type=float, default=0.2)
    parser.add_argument("--fix_grid2x4", action="store_true")
    parser.add_argument("--dynamic_stitching", action="store_true")
    parser.add_argument("--online_detector", type=str, default="gdino")
    parser.add_argument("--use_new_detections", action="store_true")
    parser.add_argument(
        "--get_bbox_index_type",
        type=str,
        default="vlm_select",
        choices=["constant", "vlm_select"],
    )

    args = parser.parse_args()

    other_configs = {
        "evaluate_function": args.evaluate_function,
        "post_process_erosion": args.post_process_erosion,
        "post_process_dilation": args.post_process_dilation,
        "kernel_size": args.kernel_size,
        "post_process_component": args.post_process_component,
        "post_process_component_num": args.post_process_component_num,
        "point_filter_nb": args.point_filter_nb,
        "point_filter_std": args.point_filter_std,
        "use_sam_huge": args.use_sam_huge,
        "matching_info_path": args.matching_info_path,
        "use_bbox_prompt": args.use_bbox_prompt,
        "use_point_prompt": args.use_point_prompt,
        "point_filter_type": args.point_filter_type,
        "point_filter_tx": args.point_filter_tx,
        "point_filter_ty": args.point_filter_ty,
        "point_filter_tz": args.point_filter_tz,
        "project_color_image": args.project_color_image,
        "use_point_prompt_num": args.use_point_prompt_num,
        "gpt_max_retry": args.gpt_max_retry,
        "gpt_max_input_images": args.gpt_max_input_images,
        "get_bbox_index_type": args.get_bbox_index_type,
        "use_all_images": args.use_all_images,
        "prompt_version": args.prompt_version,
        "use_new_detections": args.use_new_detections,
        "skip_bbox_selection_when1": args.skip_bbox_selection_when1,
        "openai_temperature": args.openai_temperature,
        "openai_top_p": args.openai_top_p,
        "image_det_confidence": args.image_det_confidence,
        "use_bbox_anno_f_gpt_select_id": args.use_bbox_anno_f_gpt_select_id,
        "fix_grid2x4": args.fix_grid2x4,
        "dynamic_stitching": args.dynamic_stitching,
        "online_detector": args.online_detector,
    }

    FROM_SCRATCH = args.from_scratch
    SEARCH_ENSEMBLE = args.search_ensemble

    assert (
        FROM_SCRATCH or SEARCH_ENSEMBLE
    ), "If not from scratch or not searching ensemble, nothing happens."

    assert args.prompt_version == 3, "Prompt version should be larger than 3. Version 1 and 2 are just for ablation study during the development phase."

    if FROM_SCRATCH:
        # * Add the date and time to the output_dir
        # the priority of resume_from > exp_name
        if args.resume_from is not None:
            args.output_dir = args.resume_from
        else:
            now = datetime.datetime.now()
            date = now.strftime("%Y-%m-%d")

            if args.exp_name is not None:
                output_dir = os.path.join(args.output_dir, f"{args.exp_name}")
                args.output_dir = output_dir
            else:
                formatted_now = now.strftime(
                    "%Y-%m-%d_%H-%M-%S"
                )  # Example format: 2024-02-26_15-30-00
                args.output_dir = os.path.join(args.output_dir, formatted_now)

        mmengine.mkdir_or_exist(args.output_dir)

    if SEARCH_ENSEMBLE:
        if not FROM_SCRATCH and args.result_path is not None:
            result_path = args.result_path  # * set this to directly doing ensemble
            args.output_dir = os.path.dirname(result_path)
            # load the params from the config file in args.output_dir
            args_yaml_path = os.path.join(args.output_dir, "args.yaml")
            resumed_args = mmengine.load(args_yaml_path)
            prompt_version = resumed_args["prompt_version"]
            gpt_version = resumed_args["openaigpt_type"]

            vg_file_path = f"outputs/query_analysis/{os.path.basename(result_path).replace(f'_{gpt_version}_promptv{prompt_version}_results.json', '.csv')}"

            args.vg_file_path = vg_file_path
        else:
            # if None, then the result path is calculated from vg_file
            # This usually comes from doing ensemble after from scratch
            result_path = f"{args.output_dir}/{os.path.basename(args.vg_file_path.split('.')[0])}_{args.openaigpt_type}_promptv{args.prompt_version}_results.json"

    # * load the args.yaml to update args
    args_yaml_path = os.path.join(args.output_dir, "args.yaml")
    if mmengine.exists(args_yaml_path):
        print(f"Loading resuming args from {args_yaml_path}.")
        resumed_args = mmengine.load(args_yaml_path)
        # * update args with resumed_args
        for key, value in resumed_args.items():
            # * if different, print
            if key in args and getattr(args, key) != value and key != "resume_from":
                print(f"Updated {key} from {getattr(args, key)} to {value}")
                setattr(args, key, value)

    # Save the parser arguments to the output_dir as a YAML file
    args_yaml_path = os.path.join(args.output_dir, "args.yaml")
    mmengine.dump(vars(args), args_yaml_path)

    # Optionally, you can print the arguments to verify them
    print("Configuration parameters:")
    print(vars(args))

    log_file = open(os.path.join(args.output_dir, "log.txt"), "a")
    sys.stdout = Tee(sys.stdout, log_file)
    print("--------------Starting a new run-----------------")
    # print time, args, and whether is from scratch
    print(f"Time: {datetime.datetime.now()}")
    print(f"Args: {vars(args)}")
    print(f"From Scratch: {FROM_SCRATCH}")

    grounder = VisualGrounder(
        scene_info_path=args.scene_info_path,
        det_info_path=args.det_info_path,
        vg_file_path=args.vg_file_path,
        output_dir=args.output_dir,
        openaigpt_type=args.openaigpt_type,
        ensemble_num=args.ensemble_num,
        **other_configs,
    )

    if FROM_SCRATCH:
        grounder.eval()  # inferenceing from scratch

    if SEARCH_ENSEMBLE:
        if not mmengine.exists(result_path):
            print(
                f"{result_path} does not exist, which mainly means the from scratch process is not complete."
            )
            exit()

        vg_file = pd.read_csv(args.vg_file_path)
        results = mmengine.load(result_path)["results"]

        ensemble_nums = [1, 2, 3, 4, 5, 6, 7]
        # remove args.ensemble_num in ensemble_nums
        if args.ensemble_num in ensemble_nums:
            ensemble_nums.remove(args.ensemble_num)

        # save the args values
        args_yaml_path = os.path.join(grounder.output_dir, "args.yaml")
        mmengine.dump(vars(args), args_yaml_path)

        for ensemble_num in ensemble_nums:
            new_result_path = result_path.replace(
                ".json", f"_matching_ks{args.kernel_size}_ensemble{ensemble_num}.json"
            )
            # if already exists, skip
            if mmengine.exists(new_result_path):
                print(f"{new_result_path} already exists. Skip.")
                continue

            grounder.ensemble_num = ensemble_num
            print(
                f"Start processing ensemble_num: {ensemble_num}, using tracking: matching."
            )
            for sample in tqdm(results):
                gpt_pred_image_id = sample["gpt_pred_image_id"]
                scene_id = sample["scene_id"]
                query = sample["query"]
                target_id = sample["target_id"]
                gpt_pred_image_id = sample["gpt_pred_image_id"]

                row = get_csv_row(vg_file, scene_id, query)
                pred_target_class = row["pred_target_class"]
                matched_image_ids = eval(
                    row[f"matched_image_ids_confidence{grounder.image_det_confidence}"]
                )
                file_prefix = f"{query.replace('/', 'or').replace(' ', '_')[:60]}"
                intermediate_output_dir = os.path.join(
                    grounder.output_dir, "intermediate_results", scene_id, file_prefix
                )

                # load the sam mask
                sam_mask_output_dir = f"{intermediate_output_dir}/sam_mask_{gpt_pred_image_id:05d}_{pred_target_class.replace('/', 'or')}_target_{target_id}"
                sam_mask_output_path = (
                    f"{sam_mask_output_dir}/anchor_sam_ks{args.kernel_size}.pkl"
                )

                if len(matched_image_ids) == 0:
                    # Use all images
                    num_posed_images = grounder.scene_infos.get_num_posed_images(
                        scene_id
                    )
                    print(
                        f"No matched images for {scene_id}: {query}. Using all {num_posed_images} images."
                    )
                    matched_image_ids = [f"{i:05d}" for i in range(num_posed_images)]

                # filter out those with invalid extrinsics, not possible that all match_image_ids are invalid
                matched_image_ids = [
                    match_image_id
                    for match_image_id in matched_image_ids
                    if grounder.scene_infos.is_posed_image_valid(
                        scene_id, match_image_id
                    )
                ]
                assert (
                    len(matched_image_ids) > 0
                ), f"All matched_image_ids are invalid in {scene_id}. This is very strange!"

                if mmengine.exists(sam_mask_output_path):
                    pred_sam_mask = mmengine.load(sam_mask_output_path)
                elif mmengine.exists(
                    sam_mask_output_path.replace(f"ks{args.kernel_size}.pkl", "raw.pkl")
                ):
                    # read mask and process from raw
                    print("process raw mask with kernel size", args.kernel_size)
                    pred_sam_mask = mmengine.load(
                        sam_mask_output_path.replace(
                            f"ks{args.kernel_size}.pkl", "raw.pkl"
                        )
                    )
                    if grounder.post_process:
                        pred_sam_mask = grounder.post_process_mask(pred_sam_mask)
                        pred_sam_mask.save(
                            f"{sam_mask_output_dir}/anchor_sam_postprocessed.jpg"
                        )
                    mmengine.dump(
                        pred_sam_mask,
                        f"{sam_mask_output_dir}/anchor_sam_ks{args.kernel_size}.pkl",
                    )
                else:
                    print("no raw mask!")
                    pred_sam_mask = None

                if grounder.use_new_detections and pred_sam_mask is None:
                    pred_bbox = None
                else:
                    pred_bbox = grounder.get_det_bbox3d_from_image_id(
                        scene_id=scene_id,
                        image_id=gpt_pred_image_id,
                        pred_target_class=pred_target_class,
                        pred_detection=None,
                        gt_object_id=target_id,
                        query=query,
                        matched_image_ids=matched_image_ids,
                        intermediate_output_dir=intermediate_output_dir,
                        bbox_index=sample.get("bbox_index", 0),
                        pred_sam_mask=pred_sam_mask,
                    )

                if pred_bbox is None:
                    if sample["eval_result"]["gpt_pred_bbox"] is not None:
                        print(
                            f"Ensemble pred_bbox is None. The original pred_bbox is {sample['eval_result']['gpt_pred_bbox']}."
                        )
                    sample["eval_result"]["iou3d"] = -1
                    sample["eval_result"]["acc_iou_25"] = False
                    sample["eval_result"]["acc_iou_50"] = False
                    sample["eval_result"]["gpt_pred_bbox"] = None
                    continue
                else:
                    eval_result = grounder.evaluate_result_func(pred_bbox, row)
                sample["eval_result"] = eval_result

            # save the new results
            final_results, is_complete = grounder.calculate_final_results(results)
            if is_complete:
                # add ensemble_num at the suffix
                mmengine.dump(final_results, new_result_path, indent=2)
            else:
                print("The results are not complete.")
