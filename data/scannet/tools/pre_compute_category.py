# This script pre-compute whether each query:
## Unique, Multi for ScanRefer
## Easy, Hard, Dep, Indep for ReferIt3D
## Codes mainly refer to BUTD-DETR

import argparse

import pandas as pd

from vlm_grounder.utils import SceneInfoHandler


def add_columns_to_csv(file_path, new_columns, functions, suffix, *args, **kwargs):
    """
    Add new columns to a CSV file based on provided functions, then save the modified DataFrame to a new CSV file.
    This version supports additional arguments for each function.

    Parameters:
    - file_path (str): The file path of the original CSV file.
    - new_columns (list of str): A list of names for the new columns to be added.
    - functions (list of callable): A list of functions to apply to each row of the DataFrame to compute values for the new columns.
    - suffix (str): A suffix to append to the original file name for saving the new CSV file.
    - args, kwargs: Additional positional and keyword arguments passed to each function.

    Returns:
    The updated new csv file. Saves a new CSV file with the added columns.
    """

    # Load the CSV file
    df = pd.read_csv(file_path)

    # Ensure the length of new_columns and functions are the same
    if len(new_columns) != len(functions):
        raise ValueError("The number of column names and functions must match")

    # Add new columns to the DataFrame
    for col_name, func in zip(new_columns, functions):
        # Apply the function to compute the new column's values
        df[col_name] = df.apply(lambda row: func(row, *args, **kwargs), axis=1)

    # Generate a new file path by appending the suffix before the .csv extension
    new_file_path = file_path.replace(".csv", f"_{suffix}.csv")

    # Save the modified DataFrame to a new CSV file
    df.to_csv(new_file_path, index=False)
    print(f"File has been saved as: {new_file_path}")

    return df


def verify_unique_easy(file_path):
    """
    Verify each row of the CSV file to ensure the following conditions:
    1. If 'is_easy_referit3d' is False, then both 'is_unique_scanrefer' and 'is_unique_category' must not be True.
    2. If either 'is_unique_scanrefer' or 'is_unique_category' is True, then 'distractor_number' must be 0.
    3. If 'is_unique_scanrefer' is True, then 'is_unique_category' must also be True.

    Parameters:
    - file_path (str): The path to the CSV file to be verified.

    Returns:
    A list of row indices that fail these verification checks.
    """

    # Load the CSV file
    if isinstance(file_path, pd.DataFrame):
        df = file_path
    else:
        df = pd.read_csv(file_path)

    # List to store the indices of rows that fail verification
    failed_indices = []

    # Iterate over each row by index and row data
    for idx, row in df.iterrows():
        # Check rule 1
        if not row["is_easy_referit3d"]:
            if row["is_unique_scanrefer"] or row["is_unique_category"]:
                failed_indices.append(idx)

        # Check rule 2
        if (row["is_unique_scanrefer"] or row["is_unique_category"]) and row[
            "distractor_number"
        ] != 0:
            failed_indices.append(idx)

        # Check rule 3
        if row["is_unique_scanrefer"] and not row["is_unique_category"]:
            failed_indices.append(idx)

    if failed_indices:
        print(f"Verification failed for the following row indices: {failed_indices}")
    else:
        print("All rows passed verification checks.")


class VGCondition:
    def __init__(self, scene_infos) -> None:
        self.scene_infos = scene_infos
        self.class2label_scannet18 = {
            "cabinet": 0,
            "bed": 1,
            "chair": 2,
            "sofa": 3,
            "table": 4,
            "door": 5,
            "window": 6,
            "bookshelf": 7,
            "picture": 8,
            "counter": 9,
            "desk": 10,
            "curtain": 11,
            "refrigerator": 12,
            "shower curtain": 13,
            "toilet": 14,
            "sink": 15,
            "bathtub": 16,
            "others": 17,
        }
        # ! note here, in the tsv file used in some codebase(like BUTE-DETR) 'refrigerator' should be 'refridgerator'
        # ! so if you use the tsv file, you should use 'refridgerator' instead of 'refrigerator' or the 'refrigerator' will be 'others'
        self.label2class_scannet18 = {
            v: k for k, v in self.class2label_scannet18.items()
        }

        self.scannet_label_mapping_file = (
            "data/scannet/meta_data/scannetv2-labels.combined.tsv"
        )
        self.scannet_label_mapping = pd.read_csv(
            self.scannet_label_mapping_file, sep="\t"
        )

        self.raw2label_scannet18 = self._get_raw2label_scannet18()
        self.raw2category = self._get_raw2category()

    def _get_raw2category(self):
        # raw means raw_category
        raw2category = pd.Series(
            self.scannet_label_mapping.category.values,
            index=self.scannet_label_mapping.raw_category,
        ).to_dict()
        return raw2category

    def _get_category_from_raw(self, raw_category):
        raw_category = " ".join(raw_category.split("_"))
        return self.raw2category[raw_category]

    # * Codes below are directly copied from ScanRefer, which uses scannet18 class
    def _get_raw2label_scannet18(self):
        scannet_labels = self.class2label_scannet18.keys()
        scannet2label = {label: i for i, label in enumerate(scannet_labels)}

        lines = [line.rstrip() for line in open(self.scannet_label_mapping_file)]
        lines = lines[1:]
        raw2label = {}
        for i in range(len(lines)):
            label_classes_set = set(scannet_labels)
            elements = lines[i].split("\t")
            raw_name = elements[1]  # raw_category
            nyu40_name = elements[7]  # nyu40class
            if nyu40_name not in label_classes_set:
                raw2label[raw_name] = scannet2label["others"]
            else:
                raw2label[raw_name] = scannet2label[nyu40_name]

        return raw2label

    def _get_label_id_from_raw_scannet18(self, raw_category):
        # First replace '_' with ' '
        raw_category = " ".join(raw_category.split("_"))
        return (
            self.raw2label_scannet18[raw_category]
            if raw_category in self.raw2label_scannet18
            else self.raw2label_scannet18["others"]
        )

    def _get_class_from_raw_scannet18(self, raw_category):
        label_id = self._get_label_id_from_raw_scannet18(raw_category)

        return self.label2class_scannet18[label_id]

    ### Based on the 'nyu40class'column in scannetv2 labels mapping file and ScanNet 18 classes
    def is_unique_scanrefer_scannet18(self, row):
        """
        In this function, "others" may be unique or multi, depending on the number of "others".
        """
        scene_id = row["scan_id"]
        target_id = row["target_id"]
        raw_category = scene_infos.get_object_raw_category(scene_id, target_id)
        raw_categories = scene_infos.get_scene_raw_categories(scene_id)

        # * need to do label mapping before checking is unique or not
        # * adopt the codes from ScanRefer codebase

        current_label_id = self._get_label_id_from_raw_scannet18(raw_category)
        scene_label_ids = [
            self._get_label_id_from_raw_scannet18(raw_category)
            for raw_category in raw_categories
        ]

        if scene_label_ids.count(current_label_id) == 1:
            return True
        else:
            return False

    def is_multi_scanrefer_scannet18(self, row):
        return not self.is_unique_scanrefer_scannet18(row)

    def get_class_from_raw_scannet18(self, row):
        scene_id = row["scan_id"]
        target_id = row["target_id"]
        raw_category = scene_infos.get_object_raw_category(scene_id, target_id)
        return self._get_class_from_raw_scannet18(raw_category)

    ### Based on the 'category' column in scannetv2 labels mapping file
    def is_unique_category(self, row):
        # judge by
        scene_id = row["scan_id"]
        target_id = row["target_id"]
        raw_category = scene_infos.get_object_raw_category(scene_id, target_id)
        raw_categories = scene_infos.get_scene_raw_categories(scene_id)

        current_category = self._get_category_from_raw(raw_category)
        scene_categories = [
            self._get_category_from_raw(raw_category) for raw_category in raw_categories
        ]

        if scene_categories.count(current_category) == 1:
            return True
        else:
            return False

    def is_multi_category(self, row):
        return not self.is_unique_category(row)

    def get_category_from_raw(self, row):
        scene_id = row["scan_id"]
        target_id = row["target_id"]
        raw_category = scene_infos.get_object_raw_category(scene_id, target_id)
        return self._get_category_from_raw(raw_category)

    ### For ReferIt3D Easy or Hard
    def decode_stimulus_string(self, stimulus_id):
        """
        Directly copied from ReferIt3D codebase
        Split into scene_id, instance_label, # objects, target object id,
        distractors object id.

        :param s: the stimulus string
        """
        s = stimulus_id
        if len(s.split("-", maxsplit=4)) == 4:
            scene_id, instance_label, n_objects, target_id = s.split("-", maxsplit=4)
            distractors_ids = ""
        else:
            scene_id, instance_label, n_objects, target_id, distractors_ids = s.split(
                "-", maxsplit=4
            )

        instance_label = instance_label.replace("_", " ")
        n_objects = int(n_objects)
        target_id = int(target_id)
        distractors_ids = [int(i) for i in distractors_ids.split("-") if i != ""]
        assert len(distractors_ids) == n_objects - 1

        return scene_id, instance_label, n_objects, target_id, distractors_ids

    def get_distractor_number(self, row):
        # need to check stimulus_id exist or not
        if "stimulus_id" in row:
            # Nr3D
            stimulus_id = row["stimulus_id"]
            n_objects = self.decode_stimulus_string(stimulus_id)[2]
            assert (
                n_objects > 1
            ), f"n_objects should be larger than 1, stimulus_id: {stimulus_id}"

            return n_objects - 1
        else:
            if "distractor_ids" in row:
                # Sr3D
                distractor_ids = eval(row["distractor_ids"])
                assert (
                    len(distractor_ids) > 0
                ), f"distractor_ids should not be empty, utterance: {row['utterance']}"
                return len(distractor_ids)  # if only one distractor, then is easy
            else:
                # ScanRefer
                # Should get the category of the target and all the object categories in the scene
                # If there are the number of objects with the same category is <= 2, then easy, else hard
                scene_id = row["scan_id"]
                target_id = row["target_id"]
                raw_category = scene_infos.get_object_raw_category(scene_id, target_id)
                raw_categories = scene_infos.get_scene_raw_categories(scene_id)

                current_category = self._get_category_from_raw(raw_category)
                scene_categories = [
                    self._get_category_from_raw(raw_category)
                    for raw_category in raw_categories
                ]

                return scene_categories.count(current_category) - 1

    def is_easy_referit3d(self, row):
        distractor_number = self.get_distractor_number(row)

        return distractor_number <= 1

    def is_hard_referit3d(self, row):
        return not self.is_easy_referit3d(row)

    ### For ReferIt3D Dep or Indep
    def get_vd_referit3d(self, row):
        rels = {
            "front",
            "behind",
            "back",
            "left",
            "right",
            "facing",
            "leftmost",
            "rightmost",
            "looking",
            "across",
        }  # a set
        tokens = eval(row["tokens"])
        return set(tokens) & rels

    def is_vd_referit3d(self, row):
        vd = self.get_vd_referit3d(row)
        vd = list(vd)
        # if any tokens in rels, then return True
        return len(vd) > 0

    def is_vid_referit3d(self, row):
        return not self.is_vd_referit3d(row)


# main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vg_file",
        type=str,
        default="data/scannet/grounding/referit3d/scanrefer_sampled_250.csv",
    )
    parser.add_argument(
        "--scene_info",
        type=str,
        default="data/scannet/scannet_instance_data/scenes_train_val_info.pkl",
    )
    args = parser.parse_args()
    input_file_path = args.vg_file

    scene_infos = SceneInfoHandler(args.scene_info)
    vg_condition = VGCondition(scene_infos)
    is_unique_scanrefer = vg_condition.is_unique_scanrefer_scannet18
    scannet18_class = vg_condition.get_class_from_raw_scannet18
    is_unique_category = vg_condition.is_unique_category
    category = vg_condition.get_category_from_raw
    is_easy_referit3d = vg_condition.is_easy_referit3d
    get_distractor_number = vg_condition.get_distractor_number
    is_vd_referit3d = vg_condition.is_vd_referit3d
    get_vd_referit3d = vg_condition.get_vd_referit3d

    # Call the function
    new_df = add_columns_to_csv(
        input_file_path,
        [
            "is_unique_scanrefer",
            "scannet18_class",
            "is_unique_category",
            "category",
            "is_easy_referit3d",
            "distractor_number",
            "is_vd_referit3d",
            "vd_referit3d",
        ],
        [
            is_unique_scanrefer,
            scannet18_class,
            is_unique_category,
            category,
            is_easy_referit3d,
            get_distractor_number,
            is_vd_referit3d,
            get_vd_referit3d,
        ],
        "relations",
    )
    verify_unique_easy(new_df)
