import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# load state and stress fields of shape (timesteps, Nq, Nr)

state = np.load("state.npy")
stress = np.load("stress.npy")

HEALTHY = 0
ATROPHIC = 1

cmap = "inferno"
theta = 2.2  # change to max value of theta

T, Nq, Nr = state.shape

q_indices, r_indices = np.meshgrid(np.arange(Nq), np.arange(Nr), indexing="ij")
qc, rc = Nq // 2, Nr // 2

# Axial to XY conversion
dq, dr = q_indices - qc, r_indices - rc
xs = dq + 0.5 * dr
ys = (np.sqrt(3) / 2) * dr

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

for ax in (ax1, ax2):
    ax.set_aspect("equal")
    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)
    ax.axis("off")

# -----------------------
# Left: Atrophy (scatter)
# -----------------------
mask_atro0 = state[0] == ATROPHIC

sc_atro_left = ax1.scatter(
    xs[mask_atro0],
    ys[mask_atro0],
    s=3,
    c="black",
)

ax1.set_title("Atrophy")

# -----------------------
# Right: Stress (imshow) + atrophy overlay
# -----------------------

# Precompute corners for pcolormesh
# xs, ys are cell centers; we need corners
dx = np.diff(xs, axis=0).mean() if Nq > 1 else 1
dy = np.diff(ys, axis=1).mean() if Nr > 1 else 1

# Compute cell corners
x_corners = xs - dx / 2
y_corners = ys - dy / 2

# We'll just add one extra row/col by padding the last cell
x_corners = np.vstack([x_corners, x_corners[-1:, :] + dx])
x_corners = np.hstack([x_corners, x_corners[:, -1:] + dx])

y_corners = np.vstack([y_corners, y_corners[-1:, :] + dy])
y_corners = np.hstack([y_corners, y_corners[:, -1:] + dy])

# Initial stress plot
stress_img0 = np.nan_to_num(stress[0], nan=0.0)

im_stress = ax2.pcolormesh(
    x_corners,
    y_corners,
    stress_img0,
    shading="auto",
    cmap=cmap,
    vmin=0,
    vmax=theta,
)

# Atrophic overlay in black (sparse scatter)
sc_atro_right = ax2.scatter(
    xs[mask_atro0],
    ys[mask_atro0],
    s=3,
    c="black",
    zorder=2,
)

ax2.set_title("Stress field")
cbar = plt.colorbar(im_stress, ax=ax2, fraction=0.046, label="Stress")

suptitle = fig.suptitle("t = 0")


def update(frame):
    mask_atro = state[frame] == ATROPHIC

    # ---- Left: atrophy scatter ----
    sc_atro_left.set_offsets(np.column_stack([xs[mask_atro], ys[mask_atro]]))

    # ---- Right: stress imshow (FAST) ----
    stress_img = np.nan_to_num(stress[frame], nan=0.0)
    im_stress.set_array(stress_img.ravel())

    # ---- Right: atrophy overlay ----
    sc_atro_right.set_offsets(np.column_stack([xs[mask_atro], ys[mask_atro]]))

    suptitle.set_text(f"t = {frame}")

    return sc_atro_left, im_stress, sc_atro_right, suptitle


anim = FuncAnimation(
    fig,
    update,
    frames=T,
    interval=10,
    blit=False,
)

plt.show()
