"""Base model interface for expo training framework."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

import openpi.shared.array_typing as at
import openpi.training.utils as training_utils


class Model(ABC):
    """Base class for policy models used in the expo training framework.

    Subclasses handle:
    - Model initialization and weight loading
    - Raw observation preprocessing (transform pipeline)
    - Action sampling (inference)
    - Output postprocessing (unnormalization)
    - Batch formatting for actor training loss
    """

    mesh: jax.sharding.Mesh
    infer_sharding: jax.sharding.NamedSharding

    @classmethod
    @abstractmethod
    def initialize(cls, *args, **kwargs) -> tuple["Model", Any, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_params(self, train_state: training_utils.TrainState) -> at.Params:
        """Extract the best available params from a train state (ema_params if available, else params)."""
        raise NotImplementedError

    @abstractmethod
    def init_target_params(self, rng: jax.random.PRNGKey, *, resume: bool = False) -> at.Params:
        """Create a separate copy of model params for the EMA target network."""
        raise NotImplementedError

    @abstractmethod
    def process_raw_inputs(
        self,
        raw_observations: Union[Dict, jnp.ndarray, np.ndarray],
        action_dim: int,
        resize_size: int,
        normalize: bool = True,
    ) -> Dict:
        """Transform raw env observations into model-ready input dict."""
        raise NotImplementedError

    @abstractmethod
    def process_transformed_outputs(
        self,
        transformed_actions: Union[jnp.ndarray, np.ndarray],
        unnormalize: bool = True,
    ) -> jnp.ndarray:
        """Unnormalize model actions back to the environment action space."""
        raise NotImplementedError

    @abstractmethod
    def sample_actions(
        self,
        transformed_inputs: Dict,
        train_state: training_utils.TrainState,
        rng: jax.random.PRNGKey,
        train: Optional[bool] = False,
        num_samples: Optional[int] = 1,
    ) -> tuple[jnp.ndarray, float]:
        """Run model inference. Returns (unpadded_actions, sample_time_ms)."""
        raise NotImplementedError

    @abstractmethod
    def sample_training_actions(
        self,
        transformed_inputs: Dict,
        train_state: training_utils.TrainState,
        rng: jax.random.PRNGKey,
        train: Optional[bool] = True,
        num_samples: Optional[int] = 1,
    ) -> tuple[jnp.ndarray, float]:
        """Run model inference for training. Returns (unpadded_actions, sample_time_ms)."""
        raise NotImplementedError

    @abstractmethod
    def prepare_batch_for_actor(self, batch: Dict) -> Any:
        """Convert a training batch into the model's expected format for loss computation."""
        raise NotImplementedError
