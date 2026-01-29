import jax.numpy as jnp
import matplotlib.pyplot as plt
states = jnp.load("states.npz")
metrics = jnp.load("metrics.npz")
metrics["0.100"].shape
for k in metrics.keys():
    plt.title(r"$A(t)/|\Omega|$ against time for different values of $\beta$ (MS)")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$A(t)/|\Omega|$")
    plt.plot(metrics[k][..., 0] / 600**2)

# Step 1: sort keys numerically
sorted_keys = sorted(metrics.keys(), key=float)

# Step 2: convert to float for x-axis
x = [float(k) for k in sorted_keys]

# Step 3: compute y-values
# Example: just use the dictionary values
y = [metrics[k][-1, 0] / 600**2 for k in sorted_keys]

# Step 4: plot
plt.plot(x, y, marker="o")
plt.xlabel(r"Value of $\beta$")
plt.ylabel(r"$A(t)$ ")
plt.title(r"Value of $A(t)$ for large $t$ against $\beta$ (MS Model)")
plt.show()