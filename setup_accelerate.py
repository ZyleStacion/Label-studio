"""
Generates accelerate_config.yaml for this machine.
Run on EVERY machine before training.

Usage:
    python setup_accelerate.py --main-ip 192.168.1.10 --num-machines 3 --machine-rank 0
    python setup_accelerate.py --main-ip 192.168.1.10 --num-machines 3 --machine-rank 1
    python setup_accelerate.py --main-ip 192.168.1.10 --num-machines 3 --machine-rank 2
"""

import argparse
import torch
import yaml


def detect_device() -> tuple[str, int]:
    """Returns (device_type, num_processes)."""
    if torch.cuda.is_available():
        return "cuda", torch.cuda.device_count()
    if torch.backends.mps.is_available():
        return "mps", 1   # Apple Silicon always has 1 MPS device
    return "cpu", 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-ip",      required=True, help="IP of the main (rank 0) machine")
    parser.add_argument("--num-machines", required=True, type=int)
    parser.add_argument("--machine-rank", required=True, type=int, help="0 for main, 1,2,... for workers")
    parser.add_argument("--port",         default=29500, type=int)
    args = parser.parse_args()

    device_type, num_proc = detect_device()
    print(f"Detected device: {device_type}  (num_processes={num_proc})")

    config = {
        "compute_environment":        "LOCAL_MACHINE",
        "distributed_type":           "MULTI_CPU" if device_type == "cpu" else "MULTI_GPU" if device_type == "cuda" else "NO",
        "num_machines":               args.num_machines,
        "machine_rank":               args.machine_rank,
        "main_process_ip":            args.main_ip,
        "main_process_port":          args.port,
        "num_processes":              args.num_machines * num_proc,
        "mixed_precision":            "no",
        "use_cpu":                    device_type == "cpu",
        "rdzv_backend":               "static",
        "same_network":               True,
    }

    with open("accelerate_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print("Saved → accelerate_config.yaml")
    print()
    if args.machine_rank == 0:
        print("You are the MAIN machine. Start training last (after all workers are ready).")
    else:
        print(f"You are worker rank {args.machine_rank}. Start training, then start the main machine.")


if __name__ == "__main__":
    main()
