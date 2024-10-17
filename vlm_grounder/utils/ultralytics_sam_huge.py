from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from ultralytics.engine.results import Results


class UltralyticsSAMHuge:
    def __init__(self, checkpoint, device="cpu"):
        sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
        # if cuda is available, default to cuda
        if torch.cuda.is_available():
            device = "cuda:0"

        sam.to(device=device)
        self.predictor = SamPredictor(sam)

    # write a function to inference
    # can call by using the instance name instance()
    def __call__(
        self,
        source,
        bboxes=None,
        points=None,
        labels=None,
        masks=None,
        multimask_output=True,
        **kwargs,
    ):
        return self.predict(
            source,
            bboxes=bboxes,
            points=points,
            labels=labels,
            masks=masks,
            multimask_output=multimask_output,
            **kwargs,
        )

    def predict(
        self,
        source: Union[str, Path, np.ndarray, Image.Image],
        bboxes,
        points,
        labels,
        masks,
        multimask_output,
        **kwargs,
    ) -> List:
        """
        Performs predictions on the given image (only 1) source using the SAM model with one bbox and many points as the prompt.

        Note that, for the original Ultralytics API, can support the same number of multiple bboxes and points, and will return multiple masks for each pair of bboxes and point. It also seems that the original Ultralytics doesn't support negative points. TODO: May verify this by reading codes.

        Args:
            source: One image. If passing np.ndarray, should be in bgr format, will convert to rgb automatically. This is to be consistent with
                ultralytics which assumes np array in bgr format.
            bboxes: Should be (1, 4), or (4, )

        Return:
            list: A list of the model predictions with only one mask.
        """
        # Check the type of source and convert them to numpy array with RGB format
        if isinstance(source, (str, Path)):
            image = cv2.imread(str(source))
            if image is None:
                raise ValueError(
                    f"Image could not be loaded from the path: {source}. Ensure the path is correct."
                )
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(source, np.ndarray):
            # Check the shape
            assert (
                len(source.shape) == 3 and source.shape[2] == 3
            ), f"The input image must have 3 dimensions (H, W, 3) but got {source.shape}."
            image = source[
                ..., ::-1
            ]  # ultraylytics assume np array in bgr format, so manually convert to rgb
        elif isinstance(source, Image.Image):
            # Convert PIL Image to numpy array
            if source.mode != "RGB":
                source = source.convert("RGB")
            image = np.array(source)
        else:
            raise ValueError(
                "Invalid source type. Must be str, Path, numpy.ndarray, or PIL.Image. Currently support one image only."
            )

        # check if points and labels are provided, should be numpy array
        if points is not None:
            points = np.array(points)
            assert (
                points.shape[1] == 2
            ), f"Points should have shape (N, 2) but got {points.shape}."

        if labels is not None:
            labels = np.array(labels)
            assert (
                labels.shape[0] == points.shape[0]
            ), f"Labels should have shape (N, ) but got {labels.shape}."

        # bboxes should be a (4, ) array specifying xyxy, or a (1, 4).
        if bboxes is not None:
            bboxes = np.array(bboxes)
            if bboxes.ndim == 1:
                bboxes = bboxes[None, :]
            assert (
                bboxes.shape[1] == 4
            ), f"Bboxes should have shape (1, 4) or (4, ) but got {bboxes.shape}."

        self.predictor.set_image(image)

        pred_masks, scores, _ = (
            self.predictor.predict(  # masks.shape is (1, H, W) when not multimask_output, else (3, H, W)
                point_coords=points,
                point_labels=labels,
                box=bboxes,  #  bbox_2d is an 1-d arrary of 4
                mask_input=masks,
                multimask_output=multimask_output,
            )
        )

        # use the mask with the highest score
        if multimask_output:
            best_mask = pred_masks[scores.argmax()][None, :]  # H, W
        else:
            best_mask = pred_masks

        best_mask_tensor = torch.from_numpy(best_mask)

        result = Results(
            orig_img=image[:, :, ::-1],  # bgr format
            names={0: "0"},
            path=source if isinstance(source, (str, Path)) else "",
            masks=best_mask_tensor,
        )

        return [result]


if __name__ == "__main__":
    checkpoint = "checkpoints/SAM/sam_vit_h_4b8939.pth"
    sam = UltralyticsSAMHuge(checkpoint)
    source = "data/scannet/posed_images/scene0645_00/00010.jpg"
    source = Image.open(source)
    bboxes = np.array([836, 17, 1004, 342])
    points = np.array([[908, 89], [900, 200]])  # should be (N, 2)
    labels = np.array([1, 0])  # should be (N, )

    result = sam(source, bboxes=bboxes, points=points, labels=labels)[0]
    result.save("test_my_sam.jpg")
    print(result)
