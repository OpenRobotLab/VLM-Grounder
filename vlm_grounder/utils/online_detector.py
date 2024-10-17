import os
import random
import time

import cv2
import mmengine
import numpy as np
import supervision as sv
from PIL.Image import Image
from requests.exceptions import ProxyError, ReadTimeout
from ultralytics import YOLO

from vlm_grounder.utils.my_gdino import GroundingDINOAPI


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 40,
    max_delay: int = 30,
    errors: tuple = (ProxyError, ReadTimeout, RuntimeError),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay

        while True:
            try:
                return func(*args, **kwargs)
            except errors as e:
                # * print the error info
                num_retries += 1
                if num_retries > max_retries:
                    print(
                        f"[DETECTOR] Encounter error of type: {type(e).__name__}, message: {e}"
                    )
                    raise Exception(
                        f"[DETECTOR] Maximum number of retries ({max_retries}) exceeded."
                    )

                print(
                    f"[DETECTOR] Retrying after {delay} seconds due to error of type: {type(e).__name__}, message: {e}"
                )
                delay *= exponential_base * (1 + jitter * random.random())
                time.sleep(min(delay, max_delay))
            except Exception as e:
                print(
                    f"[DETECTOR] Unkown error of type: {type(e).__name__}, message: {e}"
                )
                raise e

    return wrapper


class OnlineDetector:
    def __init__(
        self,
        detection_model="Grounding-DINO-1.5Pro",
        device="cuda:0",
        global_cache_dir="./outputs/global_cache/gdino_cache",
    ):
        self.device = device
        self.global_cache_dir = global_cache_dir
        mmengine.mkdir_or_exist(self.global_cache_dir)
        self.bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=2)
        if "yolo" in detection_model:
            self.detection_model = YOLO(detection_model).to(device)
            self.detect = self.detect_yolo
        elif "Grounding-DINO" in detection_model:
            self.detection_model = GroundingDINOAPI()
            self.detect = self.detect_gdino

    @retry_with_exponential_backoff
    def detect_gdino(self, image_path, category):
        """
        Detect use GDINO api
        """
        scene_id = image_path.split("/")[-2]
        image_id = os.path.basename(image_path).split(".")[0]
        cache_file_dir = os.path.join(self.global_cache_dir, scene_id, image_id)
        mmengine.mkdir_or_exist(cache_file_dir)
        if os.path.exists(
            os.path.join(cache_file_dir, f"{category.replace(' ', '_')}.pkl")
        ):
            print("[DetectGDINO] Use GDINO cached results")
            detections = mmengine.load(
                os.path.join(cache_file_dir, f"{category.replace(' ', '_')}.pkl")
            )
            return detections

        prompts = dict(image=image_path, prompt=category)
        results = self.detection_model.inference(prompts, return_mask=True)
        boxes = np.array(results["boxes"])
        categorys = np.array(results["categorys"])
        scores = np.array(results["scores"])
        masks = self.convert_PILimage_2_mask(results["masks"])
        class_id = np.zeros(categorys.shape[0], dtype=np.int64)
        if boxes.shape[0] == 0:
            print(f"[DetectGDINO] Detect nothing for {image_path}.")
            image_shape = cv2.imread(image_path).shape[0:2]
            c_sv_result = sv.Detections(
                xyxy=np.empty((0, 4)),
                mask=np.empty((0, image_shape[0], image_shape[1])),
                confidence=np.empty((0)),
                class_id=np.empty((0), dtype=np.int64),
                data={"class_name": np.empty((0), dtype=np.str_)},
            )
        else:
            c_sv_result = sv.Detections(
                xyxy=boxes,
                mask=masks,
                confidence=scores,
                class_id=class_id,
                data={"class_name": categorys},
            )

        mmengine.dump(
            c_sv_result,
            os.path.join(cache_file_dir, f"{category.replace(' ', '_')}.pkl"),
        )
        return c_sv_result

    def detect_yolo(self, image_path, category):
        """
        Detect use yolo-world model
        """
        self.detection_model.set_classes([category])
        image = cv2.imread(image_path)
        c_det_results = self.detection_model(
            [image], conf=0.01, verbose=False
        )  # * list of B images
        c_sv_result = sv.Detections.from_ultralytics(
            c_det_results[0]
        )  # * use supervsion to annotate and save the image
        return c_sv_result

    def visualize(self, image_path, sv_result, output_path):
        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence in zip(
                sv_result.data["class_name"], sv_result.confidence
            )
        ]
        det_image = Image.open(image_path)
        annotated_image = self.bounding_box_annotator.annotate(det_image, sv_result)
        annotated_image = self.label_annotator.annotate(
            annotated_image, sv_result, labels
        )

        # * save this image
        cv2.imwrite(output_path, annotated_image)

    def convert_PILimage_2_mask(self, masks):
        """
        Extract the alpha channel in the PIL.Image object as a mask
        """
        mask_data_list = []
        for mask in masks:
            mask_data = np.array(mask)[np.newaxis, :, :, -1]
            mask_data_list.append(mask_data)
        if len(mask_data_list) > 0:
            res_masks = np.concatenate(mask_data_list, axis=0).astype(bool)
        else:
            res_masks = None
        return res_masks
