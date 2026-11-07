import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils

from openpi.training import sharding as openpi_sharding


jax.distributed.initialize(
    coordinator_address=os.environ["JAX_COORDINATOR_ADDRESS"],
    num_processes=int(os.environ["JAX_NUM_PROCESSES"]),
    process_id=int(os.environ["JAX_PROCESS_ID"]),
    local_device_ids=[int(os.environ.get("JAX_LOCAL_DEVICE_ID", "0"))],
)

print(
    {
        "process_id": jax.process_index(),
        "process_count": jax.process_count(),
        "device_count": jax.device_count(),
        "local_device_count": jax.local_device_count(),
        "devices": [str(device) for device in jax.devices()],
    },
    flush=True,
)

# Exercise the same mesh and NamedSharding path used by the trainer.  The
# vector is sharded over the FSDP axis, so each host receives its local piece.
mesh = openpi_sharding.make_mesh(2)
sharding = jax.sharding.NamedSharding(
    mesh, jax.sharding.PartitionSpec(openpi_sharding.FSDP_AXIS)
)
value = jax.device_put(jnp.arange(2, dtype=jnp.int32), sharding)
value.block_until_ready()
local_sharding = jax.sharding.SingleDeviceSharding(jax.local_devices()[0])
local_piece = np.asarray(value.addressable_shards[0].data)
gathered = multihost_utils.process_allgather(local_piece, tiled=True)
local_value = jax.device_put(gathered, local_sharding)
local_value.block_until_ready()
print(
    "mesh=", mesh,
    "sharding=", value.sharding,
    "local_sharding=", local_value.sharding,
    "local_value=", jax.device_get(local_value),
    flush=True,
)
