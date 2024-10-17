import argparse

import pandas as pd


def create_subset_csv(file_path, N):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Group by 'scan_id'
    grouped = df.groupby("scan_id")

    # Randomly select N rows from each group
    subset_df = pd.DataFrame()
    for name, group in grouped:
        if len(group) <= N:
            subset_df = pd.concat([subset_df, group])
        else:
            subset_df = pd.concat([subset_df, group.sample(n=N, random_state=0)])

    # Save the new CSV file
    new_file_name = file_path.replace(".csv", f"_subset{N}.csv")
    subset_df.to_csv(new_file_name, index=False)

    print(f"Subset CSV file created: {new_file_name}")


def random_sample_subset_csv(file_path, N):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Shuffle the DataFrame
    df_shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)

    # Take the first N rows
    sample_df = df_shuffled.head(N)

    # Save the new CSV file
    new_file_name = file_path.replace(".csv", f"_sampled_{N}.csv")
    sample_df.to_csv(new_file_name, index=False)

    print(f"Sample CSV file created with {N} rows: {new_file_name}")


def get_scene_ids_from_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Get the unique scene_ids
    scene_ids = df["scan_id"].unique()

    return scene_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_file", type=str, default="data/scannet/grounding/referit3d/scanrefer.csv"
    )
    parser.add_argument("--sample_num", type=int, default=50)
    args = parser.parse_args()
    # Example usage for multiple sample sizes

    random_sample_subset_csv(args.csv_file, args.sample_num)


# Example usage
# create_subset_csv('data/scannet/grounding/referit3d/sr3d_train.csv', 4)
# create_subset_csv('data/scannet/grounding/referit3d/nr3d.csv', 1)
