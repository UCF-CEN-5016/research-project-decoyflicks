#!/usr/bin/env python3
"""
DeepSpeed distributed launcher helper.

Provides utilities to validate hostfile and GPU allocation, set up environment
variables for distributed training, initialize the communication backend
(NCCL) via DeepSpeed, and invoke a training entry point.
"""

import argparse
import importlib.util
import logging
import os
import subprocess
import sys
from typing import Dict, List, Tuple

try:
    import deepspeed
except Exception:
    deepspeed = None

try:
    import torch.distributed as dist
except Exception:
    dist = None


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("ds_launcher")


def parse_args(argv: List[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSpeed launcher helper")
    parser.add_argument("--hostfile", required=True, help="Path to hostfile (host:ngpus per line)")
    parser.add_argument("--master_addr", default=None, help="Master node address")
    parser.add_argument("--master_port", type=int, default=12345, help="Master node port")
    parser.add_argument("--num_nodes", type=int, required=True, help="Number of nodes")
    parser.add_argument("--gpus_per_node", type=int, required=True, help="GPUs per node")
    parser.add_argument("--num_gpus", type=int, default=None, help="Total number of GPUs (overrides calculation)")
    parser.add_argument("--tensor_model_parallel_size", type=int, default=None,
                        help="Tensor model parallel size (should match total GPUs if used that way)")
    parser.add_argument("--pipeline_model_parallel_size", type=int, default=1, help="Pipeline model parallel size")
    parser.add_argument("--train_module", required=True,
                        help="Python module path to training entry point (module:callable or path/to/file.py:callable)")
    parser.add_argument("--callable_name", default="main", help="Callable name in training module to invoke")
    parser.add_argument("--use_subprocess", action="store_true",
                        help="If set, invoke training via subprocess (python -m ...) instead of importing")
    return parser.parse_args(argv)


def read_hostfile(path: str) -> Dict[str, int]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Hostfile not found: {path}")
    hosts: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for idx, raw in enumerate(f):
            line = raw.strip()
            if not line:
                continue
            if ":" not in line:
                raise ValueError(f"Invalid hostfile line {idx+1}: '{raw.strip()}'. Expected format 'hostname:ngpus'")
            host, count_str = line.split(":", 1)
            host = host.strip()
            try:
                count = int(count_str.strip())
            except ValueError:
                raise ValueError(f"Invalid GPU count on line {idx+1}: '{count_str}'")
            if count < 1:
                raise ValueError(f"GPU count must be >=1 on line {idx+1}")
            hosts[host] = count
    if not hosts:
        raise ValueError("Hostfile is empty or contained only blank lines")
    return hosts


def validate_hostfile_and_gpu_allocation(hosts: Dict[str, int], expected_nodes: int, expected_gpus_per_node: int) -> Tuple[int, int]:
    actual_nodes = len(hosts)
    if actual_nodes != expected_nodes:
        raise RuntimeError(f"Hostfile node count mismatch: expected {expected_nodes}, found {actual_nodes}")
    counts = set(hosts.values())
    if len(counts) != 1:
        raise RuntimeError(f"Inconsistent GPU counts per host in hostfile: {hosts}")
    actual_gpus_per_node = next(iter(counts))
    if actual_gpus_per_node != expected_gpus_per_node:
        raise RuntimeError(f"GPUs per node mismatch: expected {expected_gpus_per_node}, found {actual_gpus_per_node}")
    total_gpus = sum(hosts.values())
    return actual_nodes, total_gpus


def compute_world_size(num_nodes: int, gpus_per_node: int, explicit_total: int = None) -> int:
    if explicit_total is not None:
        return explicit_total
    return num_nodes * gpus_per_node


def setup_env_for_distributed(master_addr: str, master_port: int, rank: int, world_size: int) -> None:
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", str(master_port))
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))


def init_nccl_with_deepspeed() -> None:
    if deepspeed is None:
        raise RuntimeError("DeepSpeed is not available. Install deepspeed to initialize the communication backend.")
    try:
        # deepspeed.init_distributed initializes torch.distributed using backend 'nccl' by default
        deepspeed.init_distributed(dist_backend="nccl")
        logger.info("DeepSpeed distributed backend initialized (nccl).")
    except Exception as exc:
        logger.error("Failed to initialize DeepSpeed distributed backend: %s", exc)
        raise


def init_torch_distributed_if_needed() -> None:
    if dist is None:
        logger.warning("torch.distributed not available in this Python environment.")
        return
    if dist.is_available() and not dist.is_initialized():
        # The backend and init method will be set by deepspeed.init_distributed usually,
        # but in case it's not, attempt a safe init if environment variables are present.
        try:
            backend = os.environ.get("DIST_BACKEND", "nccl")
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            rank = int(os.environ.get("RANK", "0"))
            init_method = os.environ.get("INIT_METHOD", None)
            if init_method:
                dist.init_process_group(backend=backend, init_method=init_method,
                                        world_size=world_size, rank=rank)
            else:
                dist.init_process_group(backend=backend, world_size=world_size, rank=rank)
            logger.info("torch.distributed initialized with backend %s", backend)
        except Exception as exc:
            logger.debug("torch.distributed initialization skipped or failed: %s", exc)


def import_and_run_callable(module_spec: str, callable_name: str, args: List[str]) -> None:
    """
    module_spec can be:
      - a module path like package.module
      - a filesystem path like /path/to/script.py
    """
    if os.path.isfile(module_spec):
        # import by path
        spec = importlib.util.spec_from_file_location("train_module", module_spec)
        module = importlib.util.module_from_spec(spec)
        loader = spec.loader
        assert loader is not None
        loader.exec_module(module)
    else:
        module = importlib.import_module(module_spec)
    if not hasattr(module, callable_name):
        raise AttributeError(f"Module '{module_spec}' has no attribute '{callable_name}'")
    entry = getattr(module, callable_name)
    if not callable(entry):
        raise TypeError(f"Attribute '{callable_name}' of module '{module_spec}' is not callable")
    # Call the entry point. It may accept args or not.
    try:
        entry_args_count = entry.__code__.co_argcount  # type: ignore[attr-defined]
    except Exception:
        entry_args_count = 0
    if entry_args_count == 0:
        entry()
    else:
        entry(args)


def run_module_in_subprocess(module_spec: str, callable_name: str, extra_args: List[str]) -> None:
    """
    Invoke the module via a subprocess: python -m module_spec or python path/to/script.py.
    The callable_name is not directly used in this mode; the module is expected to execute when run.
    """
    if os.path.isfile(module_spec):
        cmd = [sys.executable, module_spec] + extra_args
    else:
        cmd = [sys.executable, "-m", module_spec] + extra_args
    logger.info("Launching training subprocess: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Training subprocess exited with code {proc.returncode}")


def main(argv: List[str] = None) -> None:
    args = parse_args(argv)

    hosts = read_hostfile(args.hostfile)
    validate_hostfile_and_gpu_allocation(hosts, expected_nodes=args.num_nodes, expected_gpus_per_node=args.gpus_per_node)

    total_gpus = compute_world_size(args.num_nodes, args.gpus_per_node, explicit_total=args.num_gpus)
    logger.info("Total GPUs computed: %d", total_gpus)

    if args.tensor_model_parallel_size is not None:
        if args.tensor_model_parallel_size != total_gpus:
            logger.warning("tensor_model_parallel_size (%d) does not match total GPUs (%d). This may be intentional.",
                           args.tensor_model_parallel_size, total_gpus)

    # Simple rank assignment: assume the first host is master and rank 0.
    # In a full scheduler integration this would be provided by the launcher.
    master_addr = args.master_addr or next(iter(hosts.keys()))
    rank = 0
    setup_env_for_distributed(master_addr=master_addr, master_port=args.master_port, rank=rank, world_size=total_gpus)

    # Initialize communication backend
    try:
        init_nccl_with_deepspeed()
    except Exception:
        # As a fallback, try initializing torch.distributed if possible (best-effort)
        try:
            init_torch_distributed_if_needed()
        except Exception as exc:
            logger.error("Failed to initialize any distributed backend: %s", exc)
            raise

    # Invoke training entry point
    module_spec = args.train_module
    callable_name = args.callable_name
    if args.use_subprocess:
        run_module_in_subprocess(module_spec, callable_name, extra_args=[])
    else:
        import_and_run_callable(module_spec, callable_name, args=[])


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Launcher failed: %s", e)
        sys.exit(1)