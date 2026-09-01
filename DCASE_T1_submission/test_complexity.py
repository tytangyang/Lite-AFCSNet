import argparse
import importlib
import importlib.resources as pkg_resources
import json
import os
from typing import List

from complexity import get_torch_macs_memory, MAX_MACS, MAX_PARAMS_MEMORY


def check_complexity(dummy_file: str, device_ids: List[str], submission_name: str, submission_index: int):
    module_path = f"{submission_name}.{submission_name}_{submission_index}"
    try:
        api_module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(f"Could not import module: {module_path}") from exc

    model = api_module.load_model()
    inputs = api_module.load_inputs(
        [dummy_file] * len(device_ids), device_ids, model, num_workers=1
    )

    per_device = {}
    max_macs = 0
    max_params = 0

    print("\nModel Complexity Check (per device)")
    for input_tensor, device_id in zip(inputs, device_ids):
        submodel = api_module.get_model_for_device(model, device_id)
        macs, params_bytes = get_torch_macs_memory(
            submodel, input_size=input_tensor.shape
        )
        macs = int(macs)
        params_bytes = int(params_bytes)
        max_macs = max(max_macs, macs)
        max_params = max(max_params, params_bytes)
        per_device[device_id] = {"MACs": macs, "Params": params_bytes}

        status = "PASS" if macs <= MAX_MACS and params_bytes <= MAX_PARAMS_MEMORY else "FAIL"
        print(
            f"{device_id:>3} | MACs: {macs:>10,} | "
            f"Params Bytes: {params_bytes:>8,} | {status}"
        )

    output_dir = os.path.join("predictions", f"{submission_name}_{submission_index}")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "complexity.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(
            {"per_device": per_device, "max_MACs": max_macs, "max_Params": max_params},
            handle,
            indent=2,
        )
    print(f"\nSaved complexity info to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Check DCASE Task 1 model complexity.")
    parser.add_argument("--submission_name", required=True)
    parser.add_argument("--submission_index", type=int, required=True)
    args = parser.parse_args()

    device_ids = ["a", "b", "c"] + [f"s{i}" for i in range(1, 11)]
    resource_package = importlib.import_module(f"{args.submission_name}.resources")
    dummy_resource = pkg_resources.files(resource_package).joinpath("dummy.wav")
    if not dummy_resource.is_file():
        raise FileNotFoundError(f"dummy.wav not found in {args.submission_name}.resources")

    with pkg_resources.as_file(dummy_resource) as dummy_path:
        check_complexity(
            str(dummy_path), device_ids, args.submission_name, args.submission_index
        )


if __name__ == "__main__":
    main()
