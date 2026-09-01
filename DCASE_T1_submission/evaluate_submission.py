import argparse
import importlib
import importlib.resources as pkg_resources
import json
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.hub import download_url_to_file


SPLIT_URL = "https://github.com/CPJKU/dcase2024_task1_baseline/releases/download/files/"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a DCASE Task 1 submission.")
    parser.add_argument("--submission_name", required=True)
    parser.add_argument("--submission_index", type=int, required=True)
    parser.add_argument("--dev_set_dir", required=True)
    parser.add_argument("--eval_set_dir", required=True)
    return parser.parse_args()


def _find_test_csv(dataset_dir: str, resource_package_name: str) -> str:
    resource_package = importlib.import_module(resource_package_name)
    packaged = pkg_resources.files(resource_package).joinpath("test.csv")
    if packaged.is_file():
        with pkg_resources.as_file(packaged) as path:
            # The package is normally installed unpacked; return a stable copy if needed.
            stable_path = Path(path)
            if stable_path.is_file():
                return str(stable_path)

    local_candidates = [
        Path(dataset_dir) / "split_setup" / "test.csv",
        Path(dataset_dir) / "test.csv",
    ]
    for candidate in local_candidates:
        if candidate.is_file():
            return str(candidate)

    download_dir = Path(__file__).resolve().parent / resource_package_name.replace(".", "/")
    download_dir.mkdir(parents=True, exist_ok=True)
    download_path = download_dir / "test.csv"
    print(f"test.csv not found locally; downloading to {download_path}")
    download_url_to_file(SPLIT_URL + "test.csv", str(download_path))
    return str(download_path)


def load_test_split(dataset_dir: str, resource_package_name: str) -> pd.DataFrame:
    meta_csv = Path(dataset_dir) / "meta.csv"
    if not meta_csv.is_file():
        raise FileNotFoundError(f"meta.csv not found: {meta_csv}")

    test_csv = _find_test_csv(dataset_dir, resource_package_name)
    df_meta = pd.read_csv(meta_csv, sep="\t")
    df_test = pd.read_csv(test_csv, sep="\t").drop(columns=["scene_label"], errors="ignore")
    return df_test.merge(df_meta, on="filename")


def run_evaluation(args):
    module_path = f"{args.submission_name}.{args.submission_name}_{args.submission_index}"
    print(f"Importing inference module: {module_path}")
    api = importlib.import_module(module_path)

    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")

    df_test = load_test_split(args.dev_set_dir, f"{args.submission_name}.resources")
    file_paths = [os.path.join(args.dev_set_dir, name) for name in df_test["filename"]]
    device_ids = df_test["source_label"].tolist()
    scene_labels = df_test["scene_label"].tolist()

    print("Running development-test predictions ...")
    predictions, class_order = api.predict(
        file_paths=file_paths,
        device_ids=device_ids,
        model_file_path=None,
        use_cuda=use_cuda,
    )
    label_to_index = {label: index for index, label in enumerate(class_order)}
    true_labels = [label_to_index[label] for label in scene_labels]
    predicted_labels = [int(logits.argmax()) for logits in predictions]
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

    eval_csv = Path(args.eval_set_dir) / "evaluation_setup" / "fold1_test.csv"
    df_eval = pd.read_csv(eval_csv, sep="\t")
    eval_file_paths = [os.path.join(args.eval_set_dir, name) for name in df_eval["filename"]]
    eval_device_ids = df_eval["device_id"].tolist()

    print("Running evaluation-set predictions ...")
    eval_predictions, eval_class_order = api.predict(
        file_paths=eval_file_paths,
        device_ids=eval_device_ids,
        model_file_path=None,
        use_cuda=use_cuda,
    )
    if eval_class_order != class_order:
        raise RuntimeError("Class order changed between test and evaluation inference.")

    output_dir = Path("predictions") / f"{args.submission_name}_{args.submission_index}"
    output_dir.mkdir(parents=True, exist_ok=True)

    probabilities = F.softmax(torch.stack(eval_predictions), dim=1)
    labels = [class_order[index] for index in probabilities.argmax(dim=1).tolist()]
    submission = pd.DataFrame({"filename": df_eval["filename"], "scene_label": labels})
    for index, label in enumerate(class_order):
        submission[label] = probabilities[:, index].tolist()
    submission.to_csv(output_dir / "output.csv", sep="\t", index=False)

    model = api.load_model()
    torch.save(model.model.state_dict(), output_dir / "model_state_dict.pt")
    with open(output_dir / "test_accuracy.json", "w", encoding="utf-8") as handle:
        json.dump({"Test Accuracy": round(accuracy * 100, 2)}, handle, indent=2)

    print(f"\nSubmission ready: {output_dir / 'output.csv'}")


def main():
    run_evaluation(parse_args())


if __name__ == "__main__":
    main()
