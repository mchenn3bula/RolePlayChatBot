"""Select the RX 7900 XTX before invoking a project script or Python module."""

import runpy
import sys

import torch


def main():
    if not torch.version.hip or not torch.cuda.is_available():
        raise SystemExit("ROCm GPU is unavailable. Follow AMD_TRAINING.md.")
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    devices = [i for i, name in enumerate(names) if "7900 XTX" in name]
    if len(devices) != 1:
        raise SystemExit(f"Expected one RX 7900 XTX; found {names}.")
    torch.cuda.set_device(devices[0])
    print(f"Selected {names[devices[0]]} (cuda:{devices[0]}, ROCm)", flush=True)
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: amd_run.py SCRIPT [ARGS] or amd_run.py -m MODULE [ARGS]"
        )
    if sys.argv[1] == "-m":
        if len(sys.argv) < 3:
            raise SystemExit("Missing module name.")
        sys.argv = sys.argv[2:]
        runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)
    else:
        sys.argv = sys.argv[1:]
        runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
