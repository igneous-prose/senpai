#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Exercise every GPU visible to the Senpai container through PyTorch."""

import json
import platform

import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; start Docker with --gpus all on a host with "
            "NVIDIA driver 580+ and NVIDIA Container Toolkit"
        )

    devices = []
    for index in range(torch.cuda.device_count()):
        device = torch.device("cuda", index)
        matrix = torch.ones((256, 256), device=device)
        value = (matrix @ matrix)[0, 0].item()
        if value != 256:
            raise RuntimeError(f"CUDA matrix multiplication failed on {device}")

        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
            }
        )

    print(
        json.dumps(
            {
                "status": "ok",
                "platform": platform.machine(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "devices": devices,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
