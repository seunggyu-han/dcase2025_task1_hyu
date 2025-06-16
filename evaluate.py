import os
import argparse
import importlib
import importlib.resources as pkg_resources
import pandas as pd
import torch
from torch.hub import download_url_to_file

"""
submssion_name      : Chang_HYU_task1
submission_index    : 1         / 2         / 3         / 4
ckpt                : base.ckpt / mel.ckpt  / hop.ckpt  / hop_mel.ckpt

"""

# Dataset config
dataset_config = {
    "split_url": "https://github.com/CPJKU/dcase2024_task1_baseline/releases/download/files/",
    "test_split_csv": "test.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a DCASE Task 1 submission.")
    parser.add_argument("--submission_name", type=str, default="Chang_HYU_task1")
    parser.add_argument("--submission_index", type=int, required=True)
    parser.add_argument("--dev_set_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None)
    return parser.parse_args()


def download_split_file(resource_dir: str, split_name: str) -> str:
    os.makedirs(resource_dir, exist_ok=True)
    split_path = os.path.join(resource_dir, split_name)
    if not os.path.isfile(split_path):
        print(f"Downloading {split_name} to {split_path} ...")
        download_url_to_file(dataset_config["split_url"] + split_name, split_path)
    return split_path


def load_test_split(dataset_dir: str, resource_pkg: str) -> pd.DataFrame:
    meta_csv = os.path.join(dataset_dir, "meta.csv")

    try:
        with pkg_resources.path(resource_pkg, "test.csv") as test_csv_path:
            test_csv_file = str(test_csv_path)
    except FileNotFoundError:
        print("test.csv not found in package resources. Downloading ...")
        resource_dir = os.path.join(os.path.dirname(__file__), resource_pkg.replace('.', '/'), "resources")
        test_csv_file = download_split_file(resource_dir, dataset_config["test_split_csv"])

    df_meta = pd.read_csv(meta_csv, sep="\t")
    df_test = pd.read_csv(test_csv_file, sep="\t").drop(columns=["scene_label"], errors="ignore")
    df_test = df_test.merge(df_meta, on="filename")

    return df_test


def run_evaluation(args):
    # --- Load module ---
    module_path = f"{args.submission_name}.{args.submission_name}_{args.submission_index}"
    print(f"Importing inference module: {module_path}")
    api = importlib.import_module(module_path)

    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")

    # --- Load test data ---
    print("Loading test split ...")
    df_test = load_test_split(args.dev_set_dir, f"{args.submission_name}.resources")
    file_paths = [os.path.join(args.dev_set_dir, fname) for fname in df_test["filename"]]
    device_ids = df_test["source_label"].tolist()
    scene_labels = df_test["scene_label"].tolist()

    print("Running test set predictions ...")
    predictions, class_order = api.predict(
        file_paths=file_paths,
        device_ids=device_ids,
        model_file_path=args.ckpt,  # Assuming the model file is provided
        use_cuda=use_cuda
    )

    # Attach prediction results to dataframe
    pred_indices = [pred.argmax().item() for pred in predictions]
    pred_scene_labels = [class_order[i] for i in pred_indices]
    df_test["predicted_scene_label"] = pred_scene_labels
    df_test["correct"] = (df_test["scene_label"] == df_test["predicted_scene_label"]).astype(int)
    
    # Average Accuracy
    acc = df_test["correct"].mean()
    print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")

    # Class-wise Accuracy
    print("\nClass-wise accuracy:")
    for scene, a in df_test.groupby("scene_label")["correct"].mean().items():
        print(f"  {scene:20s}: {a * 100:.2f}%")

    # Device-wise Accuracy
    print("\nDevice-wise accuracy:")
    for device, a in df_test.groupby("source_label")["correct"].mean().items():
        print(f"  {device:10s}: {a * 100:.2f}%")


def main():
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
