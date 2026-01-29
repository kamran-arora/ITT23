import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Bool

from .._base import BaseModel
from .._state import SimState


class ODEModel(BaseModel):
    """
    Updated model

    gamma_type: "constant", "type2", "type3"
    I_type: "constant", "kernel"
    theta_type: "fixed", "uniform", "normal"
    use_I: True, False
    use_radial_theta: True False
    use_exp_stress: True False
    """

    gamma_type: str = eqx.field(static=True)
    I_type: str = eqx.field(static=True)
    theta_type: str = eqx.field(static=True)
    use_I: Bool = eqx.field(static=True)
    use_radial_theta: Bool = eqx.field(static=True)
    use_exp_stress: Bool = eqx.field(static=True)

    def _initialise(self, params, key):
        _key, subkey = jr.split(key)
        _grid = jnp.zeros((self.Nq, self.Nr), dtype=jnp.int8)
        if not self.use_exp_stress:
            _stress = jnp.zeros((self.Nq, self.Nr), dtype=jnp.float32)
        elif self.use_exp_stress:
            _stress = params.stress_init * jr.uniform(
                key=subkey, shape=(self.Nq, self.Nr), dtype=jnp.float32
            )
        return SimState(grid=_grid, stress=_stress, key=_key)

    @eqx.filter_jit
    def _step(self, state, params):
        _grid, _stress, _key = state.grid, state.stress, state.key
        _area = jnp.sum(_grid == 1)
        _dist_field = self._get_cartesian()

        key, _sk1, _sk2 = jr.split(_key, 3)
        _u_rnd = jr.uniform(_sk1, (self.Nq, self.Nr))

        _init_mask = self._get_init_mask(params.init_radius)
        if self.I_type == "constant":
            _alpha = params.Galpha / (self.Nq * self.Nr)
            _I_1 = jnp.exp(_alpha * _area)
        elif self.I_type == "kernel":
            _k_sum = self._get_kernel_conv(_grid, k_size=7)
            _I_1 = jnp.exp(-params.beta * _k_sum)
        elif self.I_type == "actually_constant":
            _I_1 = 1
        _new_seeds = (_grid == 0) & _init_mask & (_u_rnd < params.lambda0 * _I_1)
        _grid = jnp.where(_new_seeds, 1, _grid).astype(jnp.int8)

        _atrophic_neigh_count = self._get_neighbors_sum(
            (_grid == 1).astype(jnp.float32)
        )
        _area = jnp.sum(_grid == 1)
        if self.I_type == "constant":
            _alpha = params.Galpha / (self.Nq * self.Nr)
            _I_2 = jnp.exp(-_alpha * _area)
        elif self.I_type == "kernel":
            _k_sum = self._get_kernel_conv(_grid, k_size=7)
            _I_2 = jnp.exp(-params.beta * _k_sum)
        elif self.I_type == "actually_constant":
            _I_2 = 1

        if self.gamma_type == "constant":
            _gamma = params.gamma
        elif self.gamma_type == "type2":
            _gamma = (
                self.Nq * self.Nr * params.gamma / (1 + (self.Nq * self.Nr) - _area)
            )
        elif self.gamma_type == "type3":
            _gamma = params.gamma * _dist_field
        elif self.gamma_type == "type4":
            _gamma = params.gamma * jnp.sqrt(_dist_field)
        

        if self.use_exp_stress:
            if self.use_I:
                _generation = (
                    _stress * params.beta * _I_2 * _atrophic_neigh_count * (_grid == 0)
                )
            elif not self.use_I:
                _generation = (
                    _stress * params.beta * _atrophic_neigh_count * (_grid == 0)
                )
        elif not self.use_exp_stress:
            if self.use_I:
                _generation = params.beta * _I_2 * _atrophic_neigh_count * (_grid == 0)
            elif not self.use_I:
                _generation = params.beta * _atrophic_neigh_count * (_grid == 0)
        _diffusion = params.D_stress * self._hex_laplacian(_stress)
        _decay = _stress * -_gamma

        _stress_new = _stress + params.dt * (_decay + _generation + _diffusion)

        if self.theta_type == "fixed":
            _theta = params.theta
        elif self.theta_type == "uniform":
            _theta = jr.uniform(
                key=_sk2,
                shape=(self.Nq, self.Nr),
                minval=params.theta_min,
                maxval=params.theta_max,
            )
        elif self.theta_type == "normal":
            _theta = (
                jr.normal(key=_sk2, shape=(self.Nq, self.Nr)) * params.theta_std
                + params.theta_mean
            )
        if self.use_radial_theta:
            _failed_by_stress = (_grid == 0) & (
                _stress_new > _theta * jnp.exp(-_dist_field / params.theta_scale)
            )
        elif not self.use_radial_theta:
            _failed_by_stress = (_grid == 0) & (_stress_new > _theta)
        _grid_new = jnp.where((_grid == 1) | _failed_by_stress, 1, 0).astype(jnp.int8)

        _atrophic_neigh_count_new = self._get_neighbors_sum(
            (_grid_new == 1).astype(jnp.float32)
        )
        _is_interior = (_grid_new == 1) & (_atrophic_neigh_count_new > 5.9)
        _stress_new = jnp.where(_is_interior, 0.0, _stress_new)

        _new_state = SimState(grid=_grid_new, stress=_stress_new, key=key)
        _metrics = jnp.array([_area, jnp.sum(_stress_new)])

        return _new_state, _metrics
