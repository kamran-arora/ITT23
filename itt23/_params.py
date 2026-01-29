import equinox as eqx


class Params(eqx.Module):
    beta: float = 0.28
    theta: float = 2.2
    gamma: float = 0.006
    D_stress: float = 0.15
    lambda0: float = 3e-5
    init_radius: float = 70.0
    Galpha: float = 8.0


class ODEParams(eqx.Module):
    """
    theta: fixed threshold
    theta_min, theta_max: uniform distribution
    theta_std, theta_mean: normal distribution
    theta_scale: radial scaling i.e. theta * exp(-r/theta_scale)
    gamma: the fixed gamma/constant in more complex gamma choices
    stress_init: prefactor for U[0, 1] random initial stresses
    """

    beta: float = 0.28
    theta: float = 2.2
    theta_min: float = 1.5
    theta_max: float = 2.5
    theta_std: float = 1.0
    theta_mean: float = 0.0
    theta_scale: float = 150.0
    gamma: float = 0.006
    D_stress: float = 0.15
    lambda0: float = 3e-5
    init_radius: float = 70.0
    Galpha: float = 8.0
    dt: float = 1.0
    stress_init: float = 0.0001
