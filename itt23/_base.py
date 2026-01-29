from abc import abstractmethod
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ._state import SimState


class BaseModel(eqx.Module):
    """
    Nq, Nr number of lattice cells (axial?)
    T: timesteps
    """

    Nq: int = eqx.field(static=True)
    Nr: int = eqx.field(static=True)
    T: int = eqx.field(static=True)

    def _get_neighbors_sum(self, field: Array) -> Array:
        """Calculates sum of values in 6 hex neighbors using rolls."""
        shifts = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
        acc = jnp.zeros_like(field)
        for dq, dr in shifts:
            acc += jnp.roll(field, shift=(dq, dr), axis=(0, 1))
        return acc

    def _hex_laplacian(self, field: Array) -> Array:
        """Discrete Laplacian on a Hex grid."""
        neighbor_sum = self._get_neighbors_sum(field)
        return neighbor_sum - 6.0 * field

    def _get_init_mask(self, radius: float) -> Array:
        """Computes the circular mask for initiation."""
        q_indices, r_indices = jnp.meshgrid(
            jnp.arange(self.Nq), jnp.arange(self.Nr), indexing="ij"
        )
        qc, rc = self.Nq // 2, self.Nr // 2

        # Axial to XY conversion
        dq, dr = q_indices - qc, r_indices - rc
        x = dq + 0.5 * dr
        y = (jnp.sqrt(3) / 2) * dr

        return (x**2 + y**2) <= radius**2

    def _get_cartesian(self) -> Array:
        """
        radial distance
        """
        q_indices, r_indices = jnp.meshgrid(
            jnp.arange(self.Nq), jnp.arange(self.Nr), indexing="ij"
        )
        qc, rc = self.Nq // 2, self.Nr // 2

        # Axial to XY conversion
        dq, dr = q_indices - qc, r_indices - rc
        x = dq + 0.5 * dr
        y = (jnp.sqrt(3) / 2) * dr

        return jnp.sqrt(x**2 + y**2)

    def _get_kernel_conv(self, grid: Array, k_size: int = 5) -> Array:
        """Computes spatial kernel sum: \sum_y K(dist(x,y)) using convolution.

        Gemini did this?? I'll look at it
        """
        # Create a simple distance-based kernel (e.g., 1/dist)
        y, x = jnp.ogrid[-k_size : k_size + 1, -k_size : k_size + 1]
        dist = jnp.sqrt(x**2 + y**2)
        kernel = jnp.where(dist == 0, 0, 1.0 / (dist + 1e-5))

        # JAX convolution expects [Batch, Channel, Height, Width]
        data = grid.astype(jnp.float32)[None, None, :, :]
        kernel_jax = kernel[None, None, :, :]

        conv = jax.lax.conv_general_dilated(data, kernel_jax, (1, 1), padding="SAME")
        return jnp.squeeze(conv)

    @abstractmethod
    def _step(self, state, params) -> Tuple[SimState, Array]:
        return NotImplementedError

    @abstractmethod
    def _initialise(self, key):
        raise NotImplementedError

    def _call(self, params, key):
        _init_state = self._initialise(params, key)

        def _scan_fn(carry, t):
            _state = carry
            _new_state, _metrics = self._step(_state, params)
            return _new_state, (_new_state.grid, _metrics)

        _final_state, (_grid_history, _metrics_history) = jax.lax.scan(
            _scan_fn, _init_state, jnp.arange(self.T)
        )
        return _grid_history, _metrics_history

    def __call__(self, params, key):
        return self._call(params, key)
