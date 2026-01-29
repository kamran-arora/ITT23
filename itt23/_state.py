import equinox as eqx
from jaxtyping import Array


class SimState(eqx.Module):
    """
    grid: (Nq, Nr) Array
    stress: (Nq, Nr) Array
    """

    grid: Array
    stress: Array
    key: any
