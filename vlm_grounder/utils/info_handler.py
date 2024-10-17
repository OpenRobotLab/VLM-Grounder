# {
#   scene_id:
#   {
# 	  num_posed_images: n
# 	  intrinsic_matrix: 4 * 4 array
# 	  images_info:
# 	  {
# 		  {image_id:05d}:
# 		  {
# 			  'image_path': 'data/scannet/posed_images/{scene_id}/{image_id:05d}'
# 			  'depth_image_path': ...png
# 			  'extrinsic_matrix': 4 * 4 array
# 		  }
# 	  }
#      object_id: * N (0-indexed)
#      {
#        "aligned_bbox": numpy.array of (7, )
#        "unaligned_bbox": numpy.array of (7, )
#        "raw_category": str
#      }
#      "axis_aligned_matrix": numpy.array of (4, 4)
#      "num_objects": int
#   }
# }

import os

import cv2
import mmengine
import numpy as np
import supervision as sv
import torch
from PIL import Image
from ultralytics.engine.results import Results

from .category_judger import CategoryJudger
from .ops import project_mask_to_3d


class SceneInfoHandler:
    def __init__(self, info_path, posed_images_root="data/scannet/posed_images"):
        try:
            self.infos = mmengine.load(info_path)
            print(f"Data from {info_path} loaded successfully.")
        except Exception as e:
            print(f"Failed to load data from {info_path}: {e}")
        self.posed_images_root = posed_images_root

    def get_intrinsic_matrix(self, scene_id, image_id=None):
        return self.infos[scene_id]["intrinsic_matrix"]  # N * 4 numpy array

    def get_extrinsic_matrix(self, scene_id, image_id):
        image_id = self.convert_image_id_to_key(image_id)
        return self.infos[scene_id]["images_info"][image_id][
            "extrinsic_matrix"
        ]  # N * 4 numpy array

    def get_image_path(self, scene_id, image_id):
        image_id = self.convert_image_id_to_key(image_id)
        if image_id is None:
            return None

        return os.path.join(self.posed_images_root, scene_id, f"{image_id}.jpg")

    def get_depth_image_path(self, scene_id, image_id):
        image_id = self.convert_image_id_to_key(image_id)
        if image_id is None:
            return None

        return os.path.join(self.posed_images_root, scene_id, f"{image_id}.png")

    def convert_image_id_to_key(self, image_id):
        try:
            # try to convert image_id as int
            image_id = int(image_id)

            if image_id < 0:
                return None

            # then convert as specific format
            image_id = f"{image_id:05d}"
        except Exception as e:
            print(f"Failed to convert image_id: {image_id}: {e}")
            return None

        return image_id

    def get_world_to_axis_align_matrix(self, scene_id, image_id=None):
        return self.infos[scene_id]["axis_align_matrix"]

    def get_num_posed_images(self, scene_id):
        return self.infos[scene_id]["num_posed_images"]

    def get_num_objects(self, scene_id):
        return self.infos[scene_id]["num_objects"]

    def get_object_gt_bbox(
        self, scene_id, object_id, axis_aligned=True, with_class_id=False
    ):
        if axis_aligned:
            bbox = self.infos[scene_id][object_id]["aligned_bbox"]
        else:
            bbox = self.infos[scene_id][object_id]["unaligned_bbox"]

        if not with_class_id:
            bbox = bbox[0:-1]
        return bbox

    def get_object_raw_category(self, scene_id, object_id):
        return self.infos[scene_id][object_id]["raw_category"]

    def get_scene_raw_categories(self, scene_id):
        """
        Return a list of raw categories of all objects in the scene without deduplication.
        """
        return [
            self.get_object_raw_category(scene_id, object_id)
            for object_id in range(self.get_num_objects(scene_id))
        ]

    def project_image_to_3d_with_mask(
        self, scene_id, image_id, mask=None, with_color=False
    ):
        intrinsic_matrix = self.get_intrinsic_matrix(scene_id, image_id)
        extrinsic_matrix = self.get_extrinsic_matrix(scene_id, image_id)
        world_to_axis_align_matrix = self.get_world_to_axis_align_matrix(scene_id)
        depth_image_path = self.get_depth_image_path(scene_id, image_id)
        if with_color:
            color_image = self.get_image_path(scene_id, image_id)
        else:
            color_image = None
        points_3d = project_mask_to_3d(
            depth_image_path,
            intrinsic_matrix,
            extrinsic_matrix,
            mask,
            world_to_axis_align_matrix,
            color_image=color_image,
        )
        return points_3d

    def is_posed_image_valid(self, scene_id, image_id):
        image_id = self.convert_image_id_to_key(image_id)
        if image_id is None:
            return False
        extrinsics = self.get_extrinsic_matrix(scene_id, image_id)
        # if contains -inf or nan, then it's invalid
        if np.any(np.isinf(extrinsics)) or np.any(np.isnan(extrinsics)):
            return False
        else:
            return True


class DetInfoHandler(SceneInfoHandler):
    def __init__(self, info_path, posed_images_root="data/scannet/posed_images"):
        super().__init__(info_path, posed_images_root)
        self.category_judger = CategoryJudger()
        self.bbox_annotator = sv.BoundingBoxAnnotator()
        self.label_annotaotr = sv.LabelAnnotator()

    def get_detections(self, scene_id, image_id):
        """
        Return:
            return a supervision.detections objects
        """
        return self.infos[scene_id][self.get_image_path(scene_id, image_id)]

    def get_detections_filtered_by_score(
        self, scene_id, image_id, threshold, detections=None
    ):
        """
        Support directly pass detections rather than loading
        """
        if detections is None:
            detections = self.get_detections(scene_id, image_id)
        return detections[detections.confidence > threshold]

    def get_detections_filtered_by_class_name(
        self, scene_id, image_id, class_name, detections=None
    ):
        if detections is None:
            detections = self.get_detections(scene_id, image_id)
        detections_classes = detections.data["class_name"]
        # iterate
        bool_mask = [
            self.category_judger.is_same_category(x, class_name)
            for x in detections_classes
        ]

        return detections[bool_mask]

    def get_detections_filtered_by_bool_mask(
        self, scene_id, image_id, bool_mask, detections=None
    ):
        """
        Args:
            bool_mask: 1-d bool array with length = len(detections)
        """
        if detections is None:
            detections = self.get_detections(scene_id, image_id)
        return detections[bool_mask]

    def annotate_image_with_detections(self, scene_id, image_id, detections):
        """
        Returns:
            annotated_imamge: a PIL.Image object
        """
        image_path = self.get_image_path(scene_id, image_id)
        ori_image = Image.open(image_path)

        annotated_image = self.bbox_annotator.annotate(
            scene=ori_image, detections=detections
        )
        annotated_image = self.label_annotaotr.annotate(
            scene=annotated_image, detections=detections
        )

        return annotated_image

    def annotate_image_with_points(self, scene_id, image_id, points):
        """
        Args:
            points: ndarray with shape (N, 2). [x, y] refers to the width and height of the image.
        """
        image_path = self.get_image_path(scene_id, image_id)
        # use opencv to annotate the image
        ori_image = cv2.imread(image_path)

        for point in points:
            cv2.circle(ori_image, (int(point[0]), int(point[1])), 10, (0, 0, 255), -1)

        annotated_image = Image.fromarray(ori_image[..., [2, 1, 0]])
        return annotated_image


class ImageMaskInfoHandler(SceneInfoHandler):
    def __init__(self, info_path, mask_image_root="data/scannet/scans"):
        super().__init__(info_path)
        self.mask_image_root = mask_image_root

    def get_instance_mask(self, scene_id, image_id, target_id) -> np.ndarray:
        """
        Args:
            scene_id: str
            image_id: int
            target_id: int
        Returns:
            target_mask: np.ndarray with shape (H, W) refers to the width and height of the image.
        """
        scene_image_id = image_id * 20
        scene_image_path = os.path.join(
            self.mask_image_root, scene_id, f"instance-filt", f"{scene_image_id}.png"
        )

        mask_image = cv2.imread(scene_image_path, cv2.IMREAD_UNCHANGED)
        # instance_mask 0表示什么都没有，所以需要+1
        target_mask = np.where(mask_image == target_id + 1, 1, 0)

        return target_mask

    def apply_transparent_mask(
        self,
        scene_id,
        image_id,
        target_id,
        alpha=0.5,
        color=(0, 255, 0),
        target_mask=None,
    ):
        """
        Test method, add a transparent instance mask to the image.
        Args:
            scene_id: str
            image_id: int
            target_id: int
            alpha: float, the transparency of the mask
            color: tuple, the color of the mask
            mask: np.ndarray, the mask to apply. If None, use get_instance_mask to get the mask.

        Returns:
            result: ultralytics.engine.results.Results

        Usage example:
            result = image_mask_infos.apply_transparent_mask('scene0435_00', 40, 44)
            result.save(path)
        """
        # 读取原始图像和掩码图像
        original_image_path = self.get_image_path(scene_id, image_id)

        original_image = cv2.imread(original_image_path)
        if original_image is None:
            raise FileNotFoundError(
                f"Original image not found at path: {original_image_path}"
            )

        # 获取目标掩码信息
        if target_mask is None:
            target_mask = self.get_instance_mask(
                scene_id, image_id, target_id
            )  # bool array of (H, W)

        result = Results(
            orig_img=original_image,  # bgr format
            names={0: "0"},
            path=original_image_path,
            masks=torch.from_numpy(target_mask),
        )
        return result


if __name__ == "__main__":
    scene_infos = SceneInfoHandler(
        "data/scannet/scannet_instance_data/scenes_train_val_info_w_images.pkl"
    )
    det_infos = DetInfoHandler(
        "outputs/image_instance_detector/yolov8x-worldv2_scannet200_classes/chunk-1/detection.pkl"
    )
    image_mask_infos = ImageMaskInfoHandler(
        "data/scannet/scannet_instance_data/scenes_train_val_info_w_images.pkl"
    )
