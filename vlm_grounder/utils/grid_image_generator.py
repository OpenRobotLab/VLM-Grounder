import base64
import glob
import math
import os
import random
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def resize_image_to_GPT_size(image, detail="high"):
    max_size = 2048
    min_size = 768
    image_copy = image.copy()

    width, height = image_copy.size
    aspect_ratio = width / height

    if detail == "high":
        if max(width, height) > max_size:
            if width > height:
                width = max_size
                height = int(width / aspect_ratio)
            else:
                height = max_size
                width = int(height * aspect_ratio)

            image_copy = image_copy.resize((width, height), Image.LANCZOS)

        if min(width, height) > min_size:
            if width > height:
                height = min_size
                width = int(height * aspect_ratio)
            else:
                width = min_size
                height = int(width / aspect_ratio)

            image_copy = image_copy.resize((width, height), Image.LANCZOS)
    else:
        print("resize_image_to_GPT_size: Unsupport mode:", detail)

    return image_copy


def resize_image(image, max_size=2048):
    """
    Resize an image for the longer side to be max_size, while preserving its aspect ratio.

    Args:
        image (PIL.Image.Image): The input image to be resized.
        max_size (int, optional): The maximum size (width or height) for the resized image.
                                  Defaults to 2048.

    Returns:
        PIL.Image.Image: The resized image.
    """
    image_copy = image.copy()

    width, height = image_copy.size
    aspect_ratio = width / height

    if width > height:
        new_width = max_size
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = max_size
        new_width = int(new_height * aspect_ratio)

    resized_image = image_copy.resize((new_width, new_height), Image.LANCZOS)

    return resized_image


resize_image_with_max_size_keep_ratio = resize_image


def encode_PIL_image_to_base64(image):
    # Save the image to a bytes buffer
    buf = BytesIO()
    image.save(buf, format="JPEG")

    # Get the byte data from the buffer
    byte_data = buf.getvalue()

    # Encode the byte data to base64
    base64_str = base64.b64encode(byte_data).decode("utf-8")
    return base64_str


def create_grid_images_4x1(
    images,
    pre_resize=False,
    pre_resize_reference=512,
    annotate_id=True,
    ID_array=None,
    relative_ID_size=0.05,
    ID_color="black",
):
    """
    Create a grid of images with annotations in a 4x1 layout.
    """
    image_num = len(images)
    if pre_resize:
        images = [resize_image(img, pre_resize_reference) for img in images]
    if image_num > 32:
        random_indices = random.sample(range(image_num), 32)
        images_sampled = [images[i] for i in random_indices]
        ID_array_sampled = (
            [ID_array[i] for i in random_indices] if ID_array is not None else None
        )
    else:
        images_sampled = images
        ID_array_sampled = ID_array

    grid_images = stitch_images(
        images_sampled,
        (4, 1),
        pre_resize=pre_resize,
        pre_resize_reference=pre_resize_reference,
        annotate_id=annotate_id,
        ID_array=ID_array_sampled,
        relative_ID_size=relative_ID_size,
        ID_color=ID_color,
    )
    return grid_images


def dynamic_create_grid_images(
    images,
    fix_num=6,
    pre_resize=False,
    pre_resize_reference=512,
    annotate_id=True,
    ID_array=None,
    relative_ID_size=0.05,
    ID_color="black",
):
    image_num = len(images)
    if image_num <= 4 * fix_num:
        return stitch_images(
            images,
            (4, 1),
            pre_resize=pre_resize,
            pre_resize_reference=pre_resize_reference,
            annotate_id=annotate_id,
            ID_array=ID_array,
            relative_ID_size=relative_ID_size,
            ID_color=ID_color,
        )
    elif image_num <= 8 * fix_num:
        return stitch_images(
            images,
            (2, 4),
            pre_resize=pre_resize,
            pre_resize_reference=pre_resize_reference,
            annotate_id=annotate_id,
            ID_array=ID_array,
            relative_ID_size=relative_ID_size,
            ID_color=ID_color,
        )
    elif image_num <= 16 * fix_num:
        return stitch_images(
            images,
            (8, 2),
            pre_resize=pre_resize,
            pre_resize_reference=pre_resize_reference,
            annotate_id=annotate_id,
            ID_array=ID_array,
            relative_ID_size=relative_ID_size,
            ID_color=ID_color,
        )
    else:
        return stitch_images(
            images,
            (9, 3),
            pre_resize=pre_resize,
            pre_resize_reference=pre_resize_reference,
            annotate_id=annotate_id,
            ID_array=ID_array,
            relative_ID_size=relative_ID_size,
            ID_color=ID_color,
        )


# def dynamic_create_grid_images_fix(images, fix_num=6, pre_resize=False, pre_resize_reference=512, annotate_id=True, ID_array=None, relative_ID_size=0.05, ID_color='black'):
#     """
#     共8张，分配在 [(4, 1), (2, 4), (8, 2)]

#     """
#     image_num = len(images)
#     grid_images = []
#     if image_num <= 4*fix_num: # 直接用(4, 1)即可
#         grid_images += create_grid_images(images, (4, 1), pre_resize=pre_resize, pre_resize_reference=pre_resize_reference, annotate_id=annotate_id, ID_array=ID_array, relative_ID_size=relative_ID_size, ID_color=ID_color)
#     elif image_num <= 8*fix_num: # (4,1) 和 (2,4) 混用
#         n_8 = math.ceil((image_num-4*fix_num)/4)
#         n_4 = fix_num - n_8
#         if n_4 > 0:
#             grid_images += create_grid_images(images[:n_4*4], (4, 1), pre_resize=pre_resize, pre_resize_reference=pre_resize_reference, annotate_id=annotate_id, ID_array=ID_array[:n_4*4] if ID_array is not None else None, relative_ID_size=relative_ID_size, ID_color=ID_color)
#         if n_8 > 0:
#             grid_images += create_grid_images(images[n_4*4:], (2, 4), pre_resize=pre_resize, pre_resize_reference=pre_resize_reference, annotate_id=annotate_id, ID_array=ID_array[n_4*4:] if ID_array is not None else None, relative_ID_size=relative_ID_size, ID_color=ID_color)
#     elif image_num <= 16*fix_num:
#         n_16 = math.ceil((image_num-8*fix_num)/8) #
#         n_4_8 = fix_num - n_16
#         tmp_num = max(image_num - n_16*16, 0)
#         n_8 = math.ceil(tmp_num/4 - n_4_8)
#         n_4 = n_4_8 - n_8
#         # print(n_4, n_8, n_16)
#         if n_4 > 0:
#             grid_images += create_grid_images(images[:n_4*4], (4, 1), pre_resize=pre_resize, pre_resize_reference=pre_resize_reference, annotate_id=annotate_id, ID_array=ID_array[:n_4*4] if ID_array is not None else None, relative_ID_size=relative_ID_size, ID_color=ID_color)
#         if n_8 > 0:
#             grid_images += create_grid_images(images[n_4*4:n_4*4+n_8*8], (2, 4), pre_resize=pre_resize, pre_resize_reference=pre_resize_reference, annotate_id=annotate_id, ID_array=ID_array[n_4*4:n_4*4+n_8*8] if ID_array is not None else None, relative_ID_size=relative_ID_size, ID_color=ID_color)
#         if n_16 > 0:
#             grid_images += create_grid_images(images[n_4*4+n_8*8:], (8, 2), pre_resize=pre_resize, pre_resize_reference=pre_resize_reference, annotate_id=annotate_id, ID_array=ID_array[n_4*4+n_8*8:] if ID_array is not None else None, relative_ID_size=relative_ID_size, ID_color=ID_color)
#     else:
#         grid_images = create_grid_images(images, (9, 3), pre_resize=pre_resize, pre_resize_reference=pre_resize_reference, annotate_id=annotate_id, ID_array=ID_array, relative_ID_size=relative_ID_size, ID_color=ID_color)

#     return grid_images


def dynamic_stitch_images_fix_v2(
    images,
    fix_num=6,
    pre_resize=False,
    pre_resize_reference=512,
    annotate_id=True,
    ID_array=None,
    relative_ID_size=0.05,
    ID_color="black",
):
    """
    Stitch multiple images together in a grid layout based on the number of images.

    Args:
        images (list): A list of images to be stitched together.
        fix_num (int, optional): The number of images to be fixed in each grid layout. Defaults to 6.
        pre_resize (bool, optional): Whether to resize the images before stitching. Defaults to False.
        pre_resize_reference (int, optional): The reference size for pre-resizing the images. Defaults to 512.
        annotate_id (bool, optional): Whether to annotate the IDs on the stitched images. Defaults to True.
        ID_array (list, optional): A list of IDs corresponding to the images. Defaults to None.
        relative_ID_size (float, optional): The relative size of the ID annotation. Defaults to 0.05.
        ID_color (str, optional): The color of the ID annotation. Defaults to "black".

    Returns:
        list: A list of stitched images in a grid layout.
    """
    image_num = len(images)
    grid_images = []
    if image_num <= 4 * fix_num:  # use (4, 1)
        grid_images += stitch_images(
            images,
            (4, 1),
            pre_resize=pre_resize,
            pre_resize_reference=pre_resize_reference,
            annotate_id=annotate_id,
            ID_array=ID_array,
            relative_ID_size=relative_ID_size,
            ID_color=ID_color,
        )
    elif image_num <= 8 * fix_num:  # use (4, 1) and (2, 4)
        n_8 = math.ceil((image_num - 4 * fix_num) / 4)
        n_4 = fix_num - n_8
        if n_4 > 0:
            grid_images += stitch_images(
                images[: n_4 * 4],
                (4, 1),
                pre_resize=pre_resize,
                pre_resize_reference=pre_resize_reference,
                annotate_id=annotate_id,
                ID_array=ID_array[: n_4 * 4] if ID_array is not None else None,
                relative_ID_size=relative_ID_size,
                ID_color=ID_color,
            )
        if n_8 > 0:
            grid_images += stitch_images(
                images[n_4 * 4 :],
                (2, 4),
                pre_resize=pre_resize,
                pre_resize_reference=pre_resize_reference,
                annotate_id=annotate_id,
                ID_array=ID_array[n_4 * 4 :] if ID_array is not None else None,
                relative_ID_size=relative_ID_size,
                ID_color=ID_color,
            )
    elif image_num <= 16 * fix_num: # use (4, 1), (2, 4), (8, 2)
        n_16 = math.ceil((image_num - 8 * fix_num) / 8)  
        n_4_8 = fix_num - n_16
        tmp_num = max(image_num - n_16 * 16, 0)
        n_8 = math.ceil((tmp_num - 4 * n_4_8) / 4)
        n_4 = n_4_8 - n_8
        if n_4 > 0:
            grid_images += stitch_images(
                images[: n_4 * 4],
                (4, 1),
                pre_resize=pre_resize,
                pre_resize_reference=pre_resize_reference,
                annotate_id=annotate_id,
                ID_array=ID_array[: n_4 * 4] if ID_array is not None else None,
                relative_ID_size=relative_ID_size,
                ID_color=ID_color,
            )
        if n_8 > 0:
            grid_images += stitch_images(
                images[n_4 * 4 : n_4 * 4 + n_8 * 8],
                (2, 4),
                pre_resize=pre_resize,
                pre_resize_reference=pre_resize_reference,
                annotate_id=annotate_id,
                ID_array=ID_array[n_4 * 4 : n_4 * 4 + n_8 * 8]
                if ID_array is not None
                else None,
                relative_ID_size=relative_ID_size,
                ID_color=ID_color,
            )
        if n_16 > 0:
            grid_images += stitch_images(
                images[n_4 * 4 + n_8 * 8 :],
                (8, 2),
                pre_resize=pre_resize,
                pre_resize_reference=pre_resize_reference,
                annotate_id=annotate_id,
                ID_array=ID_array[n_4 * 4 + n_8 * 8 :]
                if ID_array is not None
                else None,
                relative_ID_size=relative_ID_size,
                ID_color=ID_color,
            )
    elif image_num <= 27 * fix_num: # use (4, 1), (2, 4), (8, 2), (9, 3)
        n_27 = math.ceil((image_num - 16 * fix_num) / 11)
        n_4_8_16 = fix_num - n_27
        tmp_num = max(image_num - n_27 * 27, 0)
        n_16 = math.ceil((tmp_num - 8 * n_4_8_16) / 8)
        n_4_8 = n_4_8_16 - n_16
        tmp_num = max(tmp_num - n_16 * 16, 0)
        n_8 = math.ceil((tmp_num - 4 * n_4_8) / 4)
        n_4 = n_4_8 - n_8
        if n_4 > 0:
            grid_images += stitch_images(
                images[: n_4 * 4],
                (4, 1),
                pre_resize=pre_resize,
                pre_resize_reference=pre_resize_reference,
                annotate_id=annotate_id,
                ID_array=ID_array[: n_4 * 4] if ID_array is not None else None,
                relative_ID_size=relative_ID_size,
                ID_color=ID_color,
            )
        if n_8 > 0:
            grid_images += stitch_images(
                images[n_4 * 4 : n_4 * 4 + n_8 * 8],
                (2, 4),
                pre_resize=pre_resize,
                pre_resize_reference=pre_resize_reference,
                annotate_id=annotate_id,
                ID_array=ID_array[n_4 * 4 : n_4 * 4 + n_8 * 8]
                if ID_array is not None
                else None,
                relative_ID_size=relative_ID_size,
                ID_color=ID_color,
            )
        if n_16 > 0:
            grid_images += stitch_images(
                images[n_4 * 4 + n_8 * 8 : n_4 * 4 + n_8 * 8 + n_16 * 16],
                (8, 2),
                pre_resize=pre_resize,
                pre_resize_reference=pre_resize_reference,
                annotate_id=annotate_id,
                ID_array=ID_array[n_4 * 4 + n_8 * 8 : n_4 * 4 + n_8 * 8 + n_16 * 16]
                if ID_array is not None
                else None,
                relative_ID_size=relative_ID_size,
                ID_color=ID_color,
            )
        if n_27 > 0:
            grid_images += stitch_images(
                images[n_4 * 4 + n_8 * 8 + n_16 * 16 :],
                (9, 3),
                pre_resize=pre_resize,
                pre_resize_reference=pre_resize_reference,
                annotate_id=annotate_id,
                ID_array=ID_array[n_4 * 4 + n_8 * 8 + n_16 * 16 :]
                if ID_array is not None
                else None,
                relative_ID_size=relative_ID_size,
                ID_color=ID_color,
            )
    else: # use more than fix_num images
        grid_images += stitch_images(
            images[: 27 * fix_num],
            (9, 3),
            pre_resize=pre_resize,
            pre_resize_reference=pre_resize_reference,
            annotate_id=annotate_id,
            ID_array=ID_array[: 27 * fix_num] if ID_array is not None else None,
            relative_ID_size=relative_ID_size,
            ID_color=ID_color,
        )
        grid_images += dynamic_stitch_images_fix_v2(
            images[27 * fix_num :],
            1,
            pre_resize,
            pre_resize_reference,
            annotate_id,
            ID_array[27 * fix_num :] if ID_array is not None else None,
            relative_ID_size,
            ID_color,
        )
    return grid_images


def stitch_images(
    images,
    grid_dims,
    pre_resize=False,
    pre_resize_reference=2048.0,
    annotate_id=True,
    ID_array=None,
    relative_ID_size=0.05,
    ID_color="black",
):
    """
    Stitch multiple images together into a grid.

    Args:
        images (List[str]): List of image paths or PIL Image objects.
        grid_dims (tuple[int, int]): Dimensions of the grid (rows, columns).
        pre_resize (bool, optional): Whether to resize images before stitching. Defaults to False.
        pre_resize_reference (float, optional): Reference size for resizing images. Defaults to 2048.0.
        annotate_id (bool, optional): Whether to annotate image IDs. Defaults to True.
        ID_array (List[int], optional): List of image IDs. Defaults to None.
        relative_ID_size (float, optional): Relative font size for image IDs. Defaults to 0.05.
        ID_color (str, optional): Color of the image ID annotation. Defaults to "black".

    Returns:
        List[Image.Image]: List of stitched grid images.
    """

    # if images[0] is str, then images should be loaded using PIL first
    if isinstance(images[0], str):
        images = [Image.open(img_path) for img_path in images]

    rows, cols = grid_dims
    if ID_array is None:
        ID_array = list(range(len(images)))
    else:
        # * assert length to be equal
        assert len(images) == len(
            ID_array
        ), "Images and ID_array should have the same length."

    # Calculate total number of images based on grid dimensions
    images_per_figure = rows * cols

    # Determine the total grid size
    total_width = images[0].size[0] * cols  # Total width of a row
    total_height = images[0].size[1] * rows  # Total height of a column
    longer_side = max(total_width, total_height)

    # Resize images if auto_resize is True
    if pre_resize and longer_side > pre_resize_reference:
        # Calculate the downsample ratio to make the longer side close to 2048 pixels
        resize_ratio = longer_side / pre_resize_reference
        images = [
            image.resize(
                (int(image.size[0] / resize_ratio), int(image.size[1] / resize_ratio))
            )
            for image in images
        ]
        # Prepare the list to hold all montage images
    grid_image_lists = []

    # Create montages
    for k in range(0, len(images), images_per_figure):
        # Extract the subset of images for current montage
        subset_images = images[k : k + images_per_figure]
        subset_ids = ID_array[k : k + images_per_figure]

        # Determine annotation font size relative to figure size
        relative_ID_size = 0.1  # Font size as a percentage of figure height

        # Determine figure size to be proportional to the number of images
        fig_width = cols * subset_images[0].size[0] / 100
        fig_height = (
            rows * subset_images[0].size[1] / 100
        )  # 100 is the default dpi of matplotlib, 1 inch = 100 pixels
        ID_size = subset_images[0].size[1] * relative_ID_size

        # Create a new figure for the montage with a smaller figure size
        fig, axs = plt.subplots(
            rows, cols, figsize=(fig_width, fig_height)
        )  # figsize is used with inch

        # Flatten the Axes array for easy iteration and indexing
        axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

        for idx, ax in enumerate(axs):
            # Remove axis for empty plots if the subset is smaller than grid size
            if idx >= len(subset_images):
                ax.axis("off")
                continue

            # Display image and remove axis
            ax.imshow(subset_images[idx])
            ax.axis("off")

            # Add annotation if required
            if annotate_id:
                ax.annotate(
                    str(subset_ids[idx]),
                    (5, 5),
                    color=ID_color,
                    fontsize=ID_size,
                    ha="left",
                    va="top",
                )

        # Adjust the layout to be tight
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

        # Save the montage figure to a bytes buffer
        buf = BytesIO()
        plt.savefig(buf, format="jpg", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)

        # Open the image for display and append to the list
        grid_image = Image.open(buf)
        grid_image_lists.append(grid_image)

    return grid_image_lists


if __name__ == "__main__":
    image_num = 80
    test_images_paths = glob.glob("data/scannet/posed_images/scene0000_00/*.jpg")
    test_images = [Image.open(path) for path in test_images_paths[:image_num]]
    ids = [os.path.basename(p) for p in test_images_paths[:image_num]]

    grid_images = dynamic_stitch_images_fix_v2(
        test_images, ID_array=ids, ID_color="red"
    )

    test_dir = "test_stitch"
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    for i, img in enumerate(grid_images):
        img.save(f"test_stitch/grid_{i}.jpg")
