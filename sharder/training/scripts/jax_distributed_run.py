"""Initialize a multi-host JAX mesh, then run an existing Python script."""

import os
import runpy
import sys

import jax


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: jax_distributed_run.py TRAINING_SCRIPT [ARGS ...]")

    jax.distributed.initialize(
        coordinator_address=os.environ["JAX_COORDINATOR_ADDRESS"],
        num_processes=int(os.environ["JAX_NUM_PROCESSES"]),
        process_id=int(os.environ["JAX_PROCESS_ID"]),
        local_device_ids=[int(os.environ.get("JAX_LOCAL_DEVICE_ID", "0"))],
    )
    print(
        "JAX distributed initialized:",
        {
            "process_id": jax.process_index(),
            "process_count": jax.process_count(),
            "device_count": jax.device_count(),
            "local_device_count": jax.local_device_count(),
            "devices": [str(device) for device in jax.devices()],
        },
        flush=True,
    )

    training_script = os.path.abspath(sys.argv[1])
    sys.argv = [training_script, *sys.argv[2:]]
    runpy.run_path(training_script, run_name="__main__")


if __name__ == "__main__":
    main()
