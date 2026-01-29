import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from .._base import BaseModel
from .._state import SimState


class OriginalModel(BaseModel):
    """
    JAX implementation of the original model given by Macular society
    """

    def _initialise(self, key):
        _grid = jnp.zeros((self.Nq, self.Nr), dtype=jnp.int8)
        _stress = jnp.zeros((self.Nq, self.Nr), dtype=jnp.float32)
        return SimState(grid=_grid, stress=_stress, key=key)

    @eqx.filter_jit
    def _step(self, state, params):
        _grid, _stress, _key = state.grid, state.stress, state.key
        _area = jnp.sum(_grid == 1)
        _alpha = params.Galpha / (self.Nq * self.Nr)
        _I = jnp.exp(-_alpha * _area)

        key, subkey = jr.split(_key)
        _u_rnd = jr.uniform(subkey, (self.Nq, self.Nr))

        _init_mask = self._get_init_mask(params.init_radius)
        _new_seeds = (_grid == 0) & _init_mask & (_u_rnd < params.lambda0 * _I)

        _atrophic_neigh_count = self._get_neighbors_sum(
            (_grid == 1).astype(jnp.float32)
        )
        _generation = params.beta * _I * _atrophic_neigh_count * (_grid == 0)
        _diffusion = params.D_stress * self._hex_laplacian(_stress)
        _decay = _stress * (1 - params.gamma)

        _stress_new = _decay + _generation + _diffusion

        _failed_by_stress = (_grid == 0) & (_stress_new > params.theta)
        _grid_new = jnp.where(
            (_grid == 1) | _new_seeds | _failed_by_stress, 1, 0
        ).astype(jnp.int8)

        _atrophic_neigh_count_new = self._get_neighbors_sum(
            (_grid_new == 1).astype(jnp.float32)
        )
        _is_interior = (_grid_new == 1) & (_atrophic_neigh_count_new > 5.9)
        _stress_new = jnp.where(_is_interior, 0.0, _stress_new)

        _new_state = SimState(grid=_grid_new, stress=_stress_new, key=key)
        _metrics = jnp.array([_area, jnp.sum(_stress_new), _I])

        return _new_state, _metrics
