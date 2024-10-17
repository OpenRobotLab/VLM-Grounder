import argparse
import os
import random
import time

import cv2
import mmengine
import numpy as np
import pandas as pd
import supervision as sv
import torch
from mmdet.apis import DetInferencer
from mmengine.utils.dl_utils import TimeCounter
from requests.exceptions import ProxyError, ReadTimeout
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results

from vlm_grounder.datasets import ScanNetPosedImages
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
                        f"[Retry] Encounter error of type: {type(e).__name__}, message: {e}"
                    )
                    raise Exception(
                        f"[Retry] Maximum number of retries ({max_retries}) exceeded."
                    )

                print(
                    f"[Retry] Retrying after {delay} seconds due to error of type: {type(e).__name__}, message: {e}"
                )
                delay *= exponential_base * (1 + jitter * random.random())
                time.sleep(min(delay, max_delay))
            except Exception as e:
                print(f"[Retry] Unkown error of type: {type(e).__name__}, message: {e}")
                raise e

    return wrapper


MM_G_DINO_CONFIG = {
    "model": "configs/mm_grounding_dino/grounding_dino_swin-l.py",
    "weights": "checkpoints/mm_grounding_dino/grounding_dino_swin-l_pretrain_all-56d69e78.pth",
}

YOLO_CONFIG = {"model": "checkpoints/yolov8_world/yolov8x-worldv2.pt"}
DEBUG = False


class ImageInstanceDetector:
    # use detection at the moment
    def __init__(
        self,
        detection_model="gdino",
        output_dir="outputs/image_instance_detector",
        csv_file=None,
        device="cuda:0",
        visualize=True,
        visualize_thresh=0.3,
        chunk_size=-1,
        visualize_mask=False,
    ):
        classes = set()
        df = pd.read_csv(csv_file)
        for i in df["pred_target_class"]:
            classes.add(i)
        self.det_classes = list(classes)
        assert len(self.det_classes) > 0, "Detect classes must be provided."
        print(self.det_classes)
        self.chunk_size = chunk_size
        self.device = device
        self.visualize = visualize
        self.visualize_mask = visualize_mask
        self.output_dir = f"{output_dir}/{os.path.basename(detection_model).split('.')[0]}_{os.path.basename(csv_file).split('.')[0]}/chunk{self.chunk_size}"
        if self.chunk_size == -1:
            # * means no chunk, then chunk_size is a big number
            self.chunk_size = 10000000
        self.chunked_det_classes = [
            self.det_classes[i : i + self.chunk_size]
            for i in range(0, len(self.det_classes), self.chunk_size)
        ]
        mmengine.mkdir_or_exist(output_dir)
        # * different detectin model
        if "yolo" in detection_model:
            self.input_image_path = False
            self.detection_model = YOLO(YOLO_CONFIG["model"]).to(device)
            self.detect = self.detect_ultralytics
        elif "Grounding-DINO" in detection_model:
            self.input_image_path = True
            self.detection_model = GroundingDINOAPI()
            self.detect = self.detect_grounding_DINO
        else:
            self.input_image_path = False
            self.detection_model = DetInferencer(**MM_G_DINO_CONFIG, device=device)
            self.detect = self.detect_mmdet

        # * supervision for annotation
        if self.visualize:
            self.bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=2)
            self.label_annotator = sv.LabelAnnotator(
                text_thickness=2, text_scale=1, text_color=sv.Color.BLACK
            )
            self.visualize_thresh = visualize_thresh
            self.visualize_output_dir = (
                f"{self.output_dir}/visualization_{self.visualize_thresh}"
            )
            mmengine.mkdir_or_exist(self.visualize_output_dir)

        # Print some basic information
        print(f"[Model] Detection model file: {detection_model}")
        print(f"[Model] CSV file: {csv_file}")
        print(f"[Model] Number of detection classes: {len(self.det_classes)}")
        print(f"[Model] Running on device: {self.device}")
        print(f"[Model] Visualization enabled: {self.visualize}")
        # BUG fix
        if self.visualize:
            print(f"[Model] Visualization threshold: {self.visualize_thresh}")
        print(f"[Model] Output directory: {self.output_dir}")
        print(f"[Model] Chunk size: {self.chunk_size}")

    def detect_ultralytics(self, images, conf_thresh=0.01):
        """
        return:
            sv_results: bboxes format in Supervision
        """
        chunked_sv_results = []
        for c_det_classes in (
            self.chunked_det_classes
        ):  # ! note, by using chunked, cannot use the class_id results anymore
            self.detection_model.set_classes(c_det_classes)
            c_det_results = self.detection_model(
                images, conf=conf_thresh, verbose=False
            )  # * list of B images
            c_sv_results = [
                sv.Detections.from_ultralytics(det_result)
                for det_result in c_det_results
            ]  # * use supervsion to annotate and save the image
            chunked_sv_results.append(c_sv_results)

        # * merge them together
        sv_results = self.merge_chunked_sv_results(chunked_sv_results)
        return sv_results

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

    @retry_with_exponential_backoff
    def detect_grounding_DINO(self, images, conf_thresh=0.01):
        chunked_sv_results = []
        for c_det_classes in self.chunked_det_classes:
            prompt = ".".join(c_det_classes)
            c_sv_results = []
            for i, image in enumerate(images):
                prompts = dict(image=image, prompt=prompt)
                results = self.detection_model.inference(prompts, return_mask=True)
                # print("grounding DINO result:", results)
                boxes = np.array(results["boxes"])
                categorys = np.array(results["categorys"])
                scores = np.array(results["scores"])
                masks = self.convert_PILimage_2_mask(results["masks"])
                class_id = np.array(
                    [self.det_classes.index(cate) for cate in categorys]
                )
                if boxes.shape[0] == 0:
                    print(f"Grounding DINO 1.5 detected nothing for {image}.")
                    image_shape = cv2.imread(image).shape[0:2]
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

                c_sv_results.append(c_sv_result)

            chunked_sv_results.append(c_sv_results)

        sv_results = self.merge_chunked_sv_results(chunked_sv_results)
        return sv_results

    def detect_mmdet(self, images, conf_thresh=0.01):
        # * every chunk_size as one list
        chunked_sv_results = []
        for c_det_classes in (
            self.chunked_det_classes
        ):  # ! note, by using chunked, cannot use the class_id results anymore
            c_det_results = self.detection_model(
                images, texts=". ".join(c_det_classes), batch_size=len(images)
            )
            c_sv_results = []
            for det_result in c_det_results["predictions"]:
                xyxy = det_result["bboxes"]
                confidence = det_result["scores"]
                class_id = det_result["labels"]
                # * need to convert to class name using class_id
                class_name = np.array([c_det_classes[id] for id in class_id])

                c_sv_result = sv.Detections(
                    xyxy=xyxy,
                    confidence=confidence,
                    class_id=class_id,
                    data={"class_name": class_name},
                )
                c_sv_results.append(c_sv_result)
            chunked_sv_results.append(c_sv_results)

        sv_results = self.merge_chunked_sv_results(chunked_sv_results)
        return sv_results

    def merge_chunked_sv_results(self, chunked_sv_results):
        # * merge them together
        sv_results = []
        for i in range(len(chunked_sv_results[0])):  # * iterate through image batch
            chunked_sv_result_i = [
                chunked_sv_result[i] for chunked_sv_result in chunked_sv_results
            ]
            sv_result_i = sv.Detections.merge(chunked_sv_result_i)
            sv_results.append(sv_result_i)

        return sv_results

    def invoke(self, image_paths, scene_ids=None, **kwargs):
        """
        images: list of images or list of image paths. If images, should be numpy array with shape (H, W, 3) and BGR format like opencv.
        """
        # * read all the images and store in images
        images = [cv2.imread(image_path) for image_path in image_paths]

        # * use detection model to detect results
        if self.input_image_path:
            sv_results = self.detect(image_paths)
        else:
            sv_results = self.detect(images)

        # * visualize
        if self.visualize:
            # * draw annotated images
            # * filter by confidence threshold
            for image, sv_result, scene_id, image_path in zip(
                images, sv_results, scene_ids, image_paths
            ):
                sv_result_filtered = sv_result[
                    sv_result.confidence > self.visualize_thresh
                ]

                labels = [
                    f"{class_name} {confidence:.2f}"
                    for class_name, confidence in zip(
                        sv_result_filtered.data.get("class_name", []),
                        sv_result_filtered.confidence,
                    )
                ]

                annotated_image = self.bounding_box_annotator.annotate(
                    image.copy(), sv_result_filtered
                )
                annotated_image = self.label_annotator.annotate(
                    annotated_image, sv_result_filtered, labels
                )

                # * save this image
                output_path = f"{self.visualize_output_dir}/{scene_id}"
                mmengine.mkdir_or_exist(output_path)
                # * save the image using its original filename with cv2
                output_filename = os.path.join(
                    output_path, os.path.basename(image_path)
                )
                cv2.imwrite(output_filename, annotated_image)

                # * save masked image if mask exist
                if sv_result_filtered.mask is not None and self.visualize_mask:
                    mask_results = []
                    mask_output_path = f"{self.visualize_output_dir}/{scene_id}/masks"
                    mmengine.mkdir_or_exist(mask_output_path)
                    for i in range(sv_result_filtered.mask.shape[0]):
                        mask = sv_result_filtered.mask[i]
                        result = Results(
                            orig_img=image,  # bgr format
                            names={"label": labels[i]},
                            path=image_path,
                            masks=torch.from_numpy(mask),
                        )
                        mask_results.append(result)
                        filename = (
                            os.path.basename(image_path).split(".")[0]
                            + f"_{labels[i].replace(' ', '_')}.jpg"
                        )
                        output_mask_filename = os.path.join(mask_output_path, filename)
                        result.save(output_mask_filename)

        return sv_results

    def save_mask_file(self, scene_id, image_id, mask):
        mask_dir = os.path.join(self.output_dir, "mask", scene_id, image_id)
        mmengine.mkdir_or_exist(mask_dir)
        mask_np = np.array(mask)
        for mask_i in range(mask_np.shape[0]):
            np.save(
                os.path.join(mask_dir, f"{mask_i}.npy"),
                np.array(np.where(mask_np[mask_i]), dtype=np.uint16),
            )

    def read_pkl(self, pkl_file="detection.pkl"):
        pkl_path = f"{self.output_dir}/{pkl_file}"
        if os.path.exists(pkl_path):
            structured_results = mmengine.load(pkl_path)
            return structured_results
        else:
            return {}

    def save_pkl(self, structured_results, pkl_file="detection.pkl"):
        output_path = f"{self.output_dir}/{pkl_file}"
        mmengine.dump(structured_results, output_path)
        # print(f"Results saved in {output_path}.")

    def format_and_save_pkl(self, results, output_file="detection.pkl"):
        structured_results = {}
        for scene_id, image_path, sv_result in results:
            if scene_id not in structured_results:
                structured_results[scene_id] = {}

            # Assuming sv_result can be converted to a serializable dictionary
            structured_results[scene_id][image_path] = sv_result

        output_path = f"{self.output_dir}/{output_file}"
        mmengine.dump(structured_results, output_path)
        # * print info
        print(f"Results saved in {output_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vg_file",
        type=str,
        default="outputs/query_analysis/prompt_v2_updated_scanrefer_sampled_50_relations.csv",
    )
    # parser.add_argument("--model", type=str, default="gdino")
    parser.add_argument("--detector", type=str, default="yolo")
    parser.add_argument("--chunk_size", type=int, default=-1)
    parser.add_argument("--visualize_thresh", type=float, default=0.2)
    args = parser.parse_args()
    model = args.detector
    chunk_size = args.chunk_size
    # if use Grounding-DINO, batch size is 1
    batch_size = 1 if model == "gdino" else 20
    output_dir = "outputs/image_instance_detector"
    save_every_batches = (
        50  #  True: Store the results after each batch of data is completed
    )
    if DEBUG:
        output_dir = "outputs/image_instance_detector/debug"
        batch_size = 1

    scene_to_process = set()
    df = pd.read_csv(args.vg_file)
    for i in df["scan_id"]:
        scene_to_process.add(i)

    image_detector = ImageInstanceDetector(
        detection_model=model,
        csv_file=args.vg_file,
        output_dir=output_dir,
        chunk_size=chunk_size,
        visualize_thresh=args.visualize_thresh,
    )

    info_file = "data/scannet/scannet_instance_data/scenes_train_val_info_w_images.pkl"
    print(f"[Running]: Processing csv file: {args.vg_file}.")
    print(f"[Running]: Using batch size {batch_size}.")

    dataset = ScanNetPosedImages(
        data_root="data/scannet/", info_file=info_file, scene_id_list=scene_to_process
    )
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    dataset_tqdm = tqdm(dataloader)

    print("dataset length:", dataset.__len__())

    # Read intermediate results
    structured_results = image_detector.read_pkl("detection.pkl")
    for batch_i, data_samples in enumerate(dataset_tqdm):
        image_paths = data_samples["image_path"]
        scene_ids = data_samples["scene_id"]
        dataset_tqdm.set_description(f"{scene_ids[0]}")
        image_paths_to_process = []
        scene_ids_to_process = []

        for i in range(len(image_paths)):
            data = structured_results.get(scene_ids[i], {}).get(image_paths[i], None)
            if data is None:
                image_paths_to_process.append(image_paths[i])
                scene_ids_to_process.append(scene_ids[i])
            else:
                print("[Skip] process", image_paths[i])
        if len(image_paths_to_process) == 0:
            print("[Skip] batch")
            continue

        with TimeCounter(tag="image_detector"):
            det_results = image_detector.invoke(
                image_paths_to_process, scene_ids_to_process
            )

        for scene_id, image_path, det_result in zip(
            scene_ids, image_paths, det_results
        ):
            if scene_id not in structured_results:
                structured_results[scene_id] = {}
            # remove mask from det_result and store in another file
            mask = det_result.mask
            det_result.mask = None
            structured_results[scene_id][image_path] = det_result
            if mask is not None:
                # store mask file
                image_id = os.path.basename(image_path).split(".")[0]
                image_detector.save_mask_file(scene_id, image_id, mask)

        # save results
        if save_every_batches > 0 and batch_i % save_every_batches == 0:
            image_detector.save_pkl(structured_results, pkl_file="detection.pkl")

    # need to save again at last
    image_detector.save_pkl(structured_results, pkl_file="detection.pkl")
