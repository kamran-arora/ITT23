# Stochastic agent based model

### Installation

1. clone this repository
2. ```pip install -e .```

### Example 1 - basic run

```Python
import itt23
import equinox as eqx
import jax.random as jr

@eqx.filter_jit
def run_simulation(model, params, key):
    return model(params, key)

model = itt23.ODEModel(
    Nq=600,
    Nr=600,
    T=1000,
    gamma_type="constant",
    I_type="constant",
    theta_type="fixed",
    use_I=True,
    use_radial_theta=False,
    use_exp_stress=False,
)
params = itt23.ODEParams()

all_states, stress_states, history = run_simulation(model, params, jr.key(123))

print(f"Final Area: {history[-1, 0]}")
print(f"Final total stress: {history[-1, 1]}")

```

### Example 2 - batch of parameters

For each value of $\beta$ it uses a different random key

```Python
import itt23
import equinox as eqx
import jax.random as jr
import jax.numpy as jnp

@eqx.filter_jit
def run_simulation(model, params, key):
    return model(params, key)

model = itt23.ODEModel(
    Nq=600,
    Nr=600,
    T=1000,
    gamma_type="constant",
    I_type="constant",
    theta_type="fixed",
    use_I=True,
    use_radial_theta=False,
    use_exp_stress=False,
)

betas = jnp.array([0.1, 0.2, 0.3, 0.4])
key = jr.key(123)
keys = jr.split(key, len(betas))

states_out = {}
stress_out = {}
metrics_out = {}

for (b, k) in zip(betas, keys):
    params = itt23.ODEParams(beta=b)
    all_states, stress_states, history = run_simulation(model, params, k)
    states_out[f"{b:.1f}"] = all_states
    stress_out[f"{b:.1f}"] = stress_states
    metrics_out[f"{b:.1f}"] = history
```