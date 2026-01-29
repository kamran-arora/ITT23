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
    gamma_type="type4",
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