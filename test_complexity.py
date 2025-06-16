import importlib
import argparse
from typing import List
from complexity import get_torch_macs_memory, MAX_MACS, MAX_PARAMS_MEMORY
import importlib.resources as pkg_resources

"""
submssion_name      : Chang_HYU_task1
submission_index    : 1         / 2         / 3         / 4
ckpt                : base.ckpt / mel.ckpt  / hop.ckpt  / hop_mel.ckpt

"""

def check_complexity(
    dummy_file: str,
    device_ids: List[str],
    submission_name: str,
    submission_index: int,
    ckpt: str = None
):
    # Dynamically import submission inference module
    module_path = f"{submission_name}.{submission_name}_{submission_index}"
    try:
        api_module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Could not import module: {module_path}") from e

    # Load model
    model = api_module.load_model(model_file_path=ckpt)
    # Create dummy input for each device
    file_paths = [dummy_file for _ in device_ids]
    inputs = api_module.load_inputs(file_paths, device_ids, model)

    # Track per-device MACs and Params
    per_device = {}
    max_macs = 0
    max_params = 0

    print("\n📊 Model Complexity Check (per device)")
    for input_tensor, device_id in zip(inputs, device_ids):
        submodel = api_module.get_model_for_device(model, device_id)
        input_shape = input_tensor.shape

        macs, params_bytes = get_torch_macs_memory(submodel, input_size=input_shape)
        max_macs = max(max_macs, macs)
        max_params = max(max_params, params_bytes)

        per_device[device_id] = {
            "MACs": macs,
            "Params": params_bytes
        }

        macs_ok = macs <= MAX_MACS
        params_ok = params_bytes <= MAX_PARAMS_MEMORY
        status = "✅" if macs_ok and params_ok else "❌"

        print(f"{device_id:>3} | MACs: {macs:>10,} | Params Bytes: {params_bytes:>8,} bytes | {status}")


def main():
    parser = argparse.ArgumentParser(description="Check model complexity for all devices using a dummy file.")
    parser.add_argument("--submission_name", type=str, default="Chang_HYU_task1")
    parser.add_argument("--submission_index", type=int, required=True)
    parser.add_argument("--ckpt", type=str, default=None)

    args = parser.parse_args()

    # Device IDs to evaluate
    device_ids = ['a', 'b', 'c'] + [f's{i}' for i in range(1, 11)]

    # Load dummy.wav from package resources
    resource_pkg = f"{args.submission_name}.resources"
    try:
        with pkg_resources.path(resource_pkg, "dummy.wav") as dummy_path:
            check_complexity(
                dummy_file=str(dummy_path),
                device_ids=device_ids,
                submission_name=args.submission_name,
                submission_index=args.submission_index,
                ckpt=args.ckpt
            )
    except FileNotFoundError:
        raise FileNotFoundError(f"'dummy.wav' not found in {resource_pkg}")


if __name__ == "__main__":
    main()
