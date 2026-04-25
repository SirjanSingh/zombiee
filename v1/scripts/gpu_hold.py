#!/usr/bin/env python
"""Reserve VRAM on a single GPU so the slot stays ours between training runs.

Allocates a contiguous tensor (default ~30 GB on a 32 GB V100) and idles in a
loop, touching the buffer once a minute so the process stays listed in
`nvidia-smi` and the GPU doesn't drop to idle clocks.

Use only when you actively plan to come back to this GPU (e.g. v2 training is
queued behind a checkpoint pull). On a shared cluster, do NOT hold more than
one GPU at a time, and release it (Ctrl-C / kill) the moment you're done.

Usage (on host with torch installed):
    CUDA_VISIBLE_DEVICES=3 python scripts/gpu_hold.py
    GPU_HOLD_GB=25 CUDA_VISIBLE_DEVICES=3 python scripts/gpu_hold.py

Usage (Docker, recommended on the DGX — uses the existing image):
    ./scripts/gpu_hold.sh 3            # hold 30 GB on GPU 3
    GB=25 ./scripts/gpu_hold.sh 3      # hold 25 GB

Stop:
    Ctrl-C if foreground, or `docker rm -f gpu-hold-N` for the docker variant.
"""

from __future__ import annotations

import datetime
import os
import sys
import time

import torch


def main():
    if not torch.cuda.is_available():
        print("ERROR: torch.cuda.is_available() is False — nothing to hold.", file=sys.stderr)
        sys.exit(1)

    gb = float(os.environ.get("GPU_HOLD_GB", "30"))
    # float32 = 4 bytes; tensor of shape (gb, 1024, 1024, 64) ~= gb GiB
    elements = int(gb * (1024 ** 3) / 4)
    # Use a 1D tensor — simplest, no shape rounding surprises.
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    free_b, total_b = torch.cuda.mem_get_info(0)
    print(
        "[{0}] device={1} cc={2}.{3} free={4:.1f}GB total={5:.1f}GB requested={6:.1f}GB pid={7}".format(
            datetime.datetime.now().isoformat(timespec="seconds"),
            name, cap[0], cap[1],
            free_b / 1e9, total_b / 1e9, gb, os.getpid(),
        )
    )
    sys.stdout.flush()

    try:
        x = torch.empty(elements, dtype=torch.float32, device="cuda:0")
    except RuntimeError as e:
        print("ERROR: allocation failed: {0}".format(e), file=sys.stderr)
        print("       try a smaller GPU_HOLD_GB (e.g. 25 or 20).", file=sys.stderr)
        sys.exit(2)
    x.fill_(0)
    torch.cuda.synchronize()

    free_after, _ = torch.cuda.mem_get_info(0)
    print(
        "[{0}] holding {1:.1f}GB; free now {2:.1f}GB. Touching every 60s.".format(
            datetime.datetime.now().isoformat(timespec="seconds"),
            x.element_size() * x.nelement() / 1e9,
            free_after / 1e9,
        )
    )
    sys.stdout.flush()

    # Heartbeat loop. add_(0) is a no-op math but keeps the process active
    # against idle reapers and shows up as compute usage in nvidia-smi.
    try:
        while True:
            x.add_(0)
            torch.cuda.synchronize()
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n[{0}] released.".format(datetime.datetime.now().isoformat(timespec="seconds")))


if __name__ == "__main__":
    main()
