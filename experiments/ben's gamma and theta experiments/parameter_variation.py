from timeit import default_timer as timer

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import itt23


@eqx.filter_jit
def run_simulation(model, params, key):
    return model(params, key)


model = itt23.ODEModel(
    Nq=600,
    Nr=600,
    T=1000,
    gamma_type="constant",
    I_type="actually_",
    theta_type="fixed",
    use_I=True,
    use_radial_theta=True,
    use_exp_stress=False,
)

betas = jnp.linspace(0.1, 10, 20)
key = jr.key(123)
keys = jr.split(key, len(betas))

states_out = {}
stress_out = {}
metrics_out = {}

for b, k in zip(betas, keys):
    params = itt23.ODEParams(beta=b)
    t0 = timer()
    # all_states, stress_states, history = run_simulation(model, params, k)
    _, _, history = run_simulation(model, params, k)
    t1 = timer()
    print(f"Time elapsed: {t1 - t0}")
    # states_out[f"{b:.3f}"] = all_states  # comment
    # stress_out[f"{b:.3f}"] = stress_states  # comment
    metrics_out[f"{b:.3f}"] = history

jnp.savez("states", **states_out)
jnp.savez("stresses", **stress_out)
jnp.savez("metrics", **metrics_out)
