# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 15:35:46 2026

@author: GM
"""

import os

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 1. PARAMETERS
# ============================================================
"""
Nq = Nr = 500          # lattice size
T = 300               # time steps
plot_interval = 5

HEALTHY = 0
ATROPHIC = 1

# Stress dynamics (KEY PARAMETERS)
beta = 0.18            # stress per atrophic neighbour
theta = 2.5            # failure threshold
gamma = 0.992          # stress retention (memory)
D_stress = 0.14        # diffusion (< 1/6)

# Gompertz inhibition
Galpha = 6.0
alpha = Galpha / (Nq * Nr)

# Stochastic initiation (early only)
lambda0 = 1.5e-5
init_radius = 60
"""
Nq = Nr = 600  # hex grid size
T = 1000  # 1000 2000 3000

HEALTHY = 0
ATROPHIC = 1
# vary parameters by *x
beta = 0.28  # 0.175*1 #0.175  1.5; vary parameters by 25% ie *0.75 or * 1.25
theta = (
    2.2  # 2.2*1.25 #2.2 1.95 1.85 1.75 1.35; vary parameters by 25% ie *0.75 or * 1.25
)
gamma = 0.006  # 0.992*1 # gamma vary by 1% & 2% *0.98 * 1.02
D_stress = 0.15 * 1  # 0.07 must be <1/6 : 6 surrounding cells in hex lattice

lambda0 = 3e-5  # 1.0*15e-6 #7e-6 4e-6
init_radius = 70  # 70*1.25

Galpha = 8  # 6*1                    # Gompertz parameter
alpha = Galpha / (Nq * Nr)  # 6 1.6 2 4 Gompertz capacity

plot_interval = 100
# ============================================================
# 2. HEX GEOMETRY
# ============================================================

hex_neighbours = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]


def axial_to_xy(q, r):
    x = q + 0.5 * r
    y = (np.sqrt(3) / 2) * r
    return x, y


def hex_laplacian(Z):
    lap = -6.0 * Z
    for dq, dr in hex_neighbours:
        lap += np.roll(np.roll(Z, dq, axis=0), dr, axis=1)
    return lap


# ============================================================
# 3. FIELDS
# ============================================================

state = np.zeros((Nq, Nr), dtype=np.int8)
stress = np.zeros((Nq, Nr), dtype=np.float32)

area_history = []
stress_history = []
results = []
parameters = []

parameters.append([Nq, T, beta, theta, gamma, D_stress, lambda0, init_radius, Galpha])


# ============================================================
# 4. CENTRAL INITIATION MASK
# ============================================================

Q, R = np.meshgrid(np.arange(Nq), np.arange(Nr), indexing="ij")
qc, rc = Nq // 2, Nr // 2
Xc, Yc = axial_to_xy(Q - qc, R - rc)
init_mask = (Xc**2 + Yc**2) <= init_radius**2

xs, ys = axial_to_xy(Q - qc, R - rc)

cmap = plt.cm.inferno.copy()
cmap.set_bad(color="black")
# ============================================================
# 5. MAIN TIME LOOP
# ============================================================

for t in range(T):
    # --------------------------------------------------------
    # Global Gompertz capacity
    # --------------------------------------------------------
    area = np.sum(state == ATROPHIC)
    I = np.exp(-alpha * area)

    # --------------------------------------------------------
    # Stochastic initiation (early seeding only)
    # --------------------------------------------------------
    rnd = np.random.rand(Nq, Nr)
    state[(state == HEALTHY) & init_mask & (rnd < lambda0 * I)] = ATROPHIC

    # -------------------------------------------------------
    # Combined update
    # -------------------------------------------------------
    _middle_term = np.zeros_like(stress)
    for dq, dr in hex_neighbours:
        neigh = np.roll(np.roll(state, dq, axis=0), dr, axis=1)
        _middle_term += beta * I * ((state == HEALTHY) & (neigh == ATROPHIC))

    stress = stress * (1 - gamma) + _middle_term + D_stress * hex_laplacian(stress)

    # --------------------------------------------------------
    # Stress decay (implicit memory)
    # --------------------------------------------------------
    # stress *= gamma

    # --------------------------------------------------------
    # Edge-driven stress generation
    # --------------------------------------------------------
    # for dq, dr in hex_neighbours:
    #     neigh = np.roll(np.roll(state, dq, axis=0), dr, axis=1)
    #     stress += beta * I * ((state == HEALTHY) & (neigh == ATROPHIC))

    # --------------------------------------------------------
    # Diffusion
    # --------------------------------------------------------
    # stress += D_stress * hex_laplacian(stress)

    # Dead cells carry no stress
    # stress[state == ATROPHIC] = 0.0
    # Zero stress only in fully surrounded (interior) atrophic cells
    atrophic_neighbors = np.zeros_like(state, dtype=np.int8)

    for dq, dr in hex_neighbours:
        neigh = np.roll(np.roll(state, dq, axis=0), dr, axis=1)
        atrophic_neighbors += neigh == ATROPHIC

    interior = (state == ATROPHIC) & (atrophic_neighbors == 6)
    stress[interior] = 0.0

    # --------------------------------------------------------
    # Deterministic failure (NO PROBABILITY, NO τ)
    # --------------------------------------------------------
    state[(state == HEALTHY) & (stress > theta)] = ATROPHIC

    # --------------------------------------------------------
    # Record
    # --------------------------------------------------------
    area_history.append(area)
    stress_history.append(stress.sum())

    results.append(
        [
            t,
            area,
            I,
            stress.sum(),
        ]
    )
    # --------------------------------------------------------
    # Visualisation
    # --------------------------------------------------------
    if np.isnan(stress).any():
        print("NaNs detected at time", t)

    if t % plot_interval == 0 or t == T - 1:
        fig = plt.figure(figsize=(12, 4))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2])

        # --- Atrophy map ---
        ax1 = fig.add_subplot(gs[0])
        ax1.scatter(xs[state == ATROPHIC], ys[state == ATROPHIC], s=3, c="black")
        ax1.set_title("Atrophy")
        ax1.set_aspect("equal")
        ax1.set_xlim(-200, 200)
        ax1.set_ylim(-200, 200)
        ax1.axis("off")

        # --- Stress field ---
        ax2 = fig.add_subplot(gs[1])
        stress_plot = np.nan_to_num(
            stress[state == HEALTHY], nan=0.0, posinf=0.0, neginf=0.0
        )

        ax2.scatter(  # include this first as black background to avoid white holes for zeros
            xs[state == ATROPHIC], ys[state == ATROPHIC], s=3, c="black", zorder=1
        )

        sc = ax2.scatter(
            xs[state == HEALTHY],  # stress field plot
            ys[state == HEALTHY],
            c=stress_plot,  # stress[state == HEALTHY],
            s=4,  # 3,
            cmap=cmap,  # "inferno",
            vmin=0,
            vmax=theta,
            zorder=2,
        )
        ax2.set_title("Stress field")
        plt.gca().set_aspect("equal")
        ax2.set_aspect("equal")
        ax2.set_xlim(-200, 200)
        ax2.set_ylim(-200, 200)
        ax2.axis("off")
        plt.colorbar(sc, ax=ax2, fraction=0.046, label="Stress")

        # --- Area & stress vs time ---
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(area_history, "k", label="Atrophic area")
        ax3.set_xlim(0, T)
        ax3.set_ylim(0, 25000)
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Area")

        ax4 = ax3.twinx()
        ax4.plot(stress_history, "b", alpha=0.7, label="Total stress")
        ax4.set_ylim(0, 25000)
        ax4.set_ylabel("Stress", color="b")

        fig.suptitle(f"t = {t}, area = {area}")
        plt.tight_layout()
        plt.show()

print(area_history[-1])


# ----------------------------
# Save CSV
# ----------------------------
results = np.array(results)
parameters = np.array(parameters)

# outdir = r"C:\temp"
# os.makedirs(outdir, exist_ok=True)

# outfile = os.path.join(outdir, "GA_sim16b_results.csv")
# outfile_p = os.path.join(outdir, "GA_sim16b_params.csv")
# #outfile_q = os.path.join(outdir, "GA_Gomp_test.csv")
# np.savetxt(
#     outfile,
#     results,
#     delimiter=",",
#     header="t,Area,I,stress",
#     comments=""
# )
# np.savetxt(
#     outfile_p,
#     parameters,
#     delimiter=",",
#     header="Nq,T,beta,theta,gamma,D_stress,lambda0,init_radius,Galpha",
#     comments=""
# )

# print("\nSaved to:", outfile)
# print("\nSaved to:", outfile_p)
# print()


# ============================================================
# END
# ============================================================
