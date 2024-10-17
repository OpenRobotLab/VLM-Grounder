class AccuracyCalculator:
    def __init__(self):
        self.results = {}

    def compute_accuracy(self, data_list):
        """
        Efficiently calculate accuracies for all 'acc_' types overall and within 'is_' type groups.
        data_list[0]['eval_result'] should never be None

        :param data_list: List of dictionaries containing evaluation results.
        """

        self.results = {}  # clear

        if len(data_list) == 0:
            return None, None

        # Determine the 'acc_' and 'is_' keys from the first item
        eval_result = data_list[0]["eval_result"]
        acc_keys = [key for key in eval_result if key.startswith("acc_")]
        is_keys = [key for key in eval_result if key.startswith("is_")]

        # Initialize nested dictionaries for each 'acc_' and 'is_' type
        for acc in acc_keys:
            self.results[acc] = {"Overall": {"total": 0, "correct": 0}}
            for is_key in is_keys:
                self.results[acc][is_key] = {"total": 0, "correct": 0}

        # Accumulate correct and total counts
        for data in data_list:
            eval_result = data["eval_result"]
            for acc in acc_keys:
                self.results[acc]["Overall"]["total"] += 1
                if eval_result[acc]:
                    self.results[acc]["Overall"]["correct"] += 1

                for is_key in is_keys:
                    if eval_result[is_key]:
                        self.results[acc][is_key]["total"] += 1
                        if eval_result[acc]:
                            self.results[acc][is_key]["correct"] += 1

        # Calculate accuracies and update the dictionary
        for acc in acc_keys:
            for key in self.results[acc]:
                total = self.results[acc][key]["total"]
                correct = self.results[acc][key]["correct"]
                if total > 0:
                    self.results[acc][key]["accuracy"] = correct / total * 100.0
                else:
                    self.results[acc][key]["accuracy"] = None

        summary = self.get_summary()

        return self.results, summary

    def get_summary(self):
        """
        Returns a summary of accuracies in a structured format for easy copying to Excel.
        """
        is_keys_order = [
            "is_unique_scanrefer",
            "is_multi_scanrefer",
            "is_easy_referit3d",
            "is_hard_referit3d",
            "is_vd_referit3d",
            "is_vid_referit3d",
        ]
        acc_keys_order = ["acc_iou_25", "acc_iou_50"]

        # Print header
        header = ["Overall"] + is_keys_order
        header = [
            item for sub in header for item in [sub + "@0.25", sub + "@0.5"]
        ]  # Expand and label with accuracies
        print("\t".join(header))

        # Print rows
        row = []
        for group in header:
            group_name, acc_suffix = group.rsplit("@", 1)
            acc_key = "acc_iou_25" if acc_suffix == "0.25" else "acc_iou_50"
            accuracy = self.results[acc_key].get(group_name, {}).get("accuracy")
            row.append(f"{accuracy:.2f}" if accuracy is not None else " ")

        print("\t".join(row))

        return row

    def print_statistics(self):
        """
        Prints the computed accuracy statistics in a clear format.
        """
        for acc in sorted(
            self.results.keys()
        ):  # Sort to ensure 'acc_' keys are printed first
            print(f"Accuracy Type: {acc}")
            for key in self.results[acc].keys():
                accuracy = self.results[acc][key]["accuracy"]
                if accuracy is not None:  # Ensures we only print computed accuracies
                    print(f"  Group: {key} - Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    # Example Usage
    import mmengine

    data_list = mmengine.load(
        "outputs/visual_grounding_video/2024-04-29_19-10-22_v1_3_erosion_k_3/updated_scanrefer_test_sampled_1000_relations_with_images_selected_and_pkl_top250_gpt-4-turbo-2024-04-09_promptv1_results_yolo_ensemble3.json"
    )["results"]

    calculator = AccuracyCalculator()
    acc_statics = calculator.compute_accuracy(data_list)
    print(acc_statics)
