import matplotlib.pyplot as plt
import numpy as np
from numpy import linspace
import os
import struct

OUT_DIR = "artifacts"
METHODS = ["NGP", "CIC", "TSC"]

def plot_particles(method, params_path  , positions_path, status_path):
    # open params file
    fp = open(params_path, "r")
    data = fp.readlines()
    fp.close()
    
    # parse params file
    data = [line.split("=") for line in data if "=" in line and not line.strip().startswith("#")]
    for line in data:
        if "NgridX" in line[0]:
            NgridX = int(line[1].split("#")[0].strip())
        if "NgridY" in line[0]:
            NgridY = int(line[1].split("#")[0].strip())
        if "BoxSizeX" in line[0]:
            BoxSizeX = float(line[1].split("#")[0].strip())
        if "BoxSizeY" in line[0]:
            BoxSizeY = float(line[1].split("#")[0].strip())
            
    print(f"NgridX: {NgridX}, NgridY: {NgridY}, BoxSizeX: {BoxSizeX}, BoxSizeY: {BoxSizeY}")

    

    # check if the files exist
    if not os.path.exists(positions_path):
        print(f"File {positions_path} does not exist")
        return
    if not os.path.exists(status_path):
        print(f"File {status_path} does not exist")
        return

    with open(positions_path, "rb") as fp:
        data = fp.read()

    # Parse binary data as doubles (8 bytes each)
    num_doubles = len(data) // 8
    positions = struct.unpack(f'{num_doubles}d', data)
    
    # Reshape into (x, y) pairs
    if len(positions) % 2 != 0:
        print(f"File {positions_path} has odd number of values, cannot form (x, y) pairs")
        return
    
    x = [positions[i] for i in range(0, len(positions), 2)]
    y = [positions[i] for i in range(1, len(positions), 2)]

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(x, y, s=5, c='black')
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    ax.set_title(f"Particle Mesh - {method}")
    
    dx = BoxSizeX / NgridX
    dy = BoxSizeY / NgridY

    ax.set_xticks(np.arange(0, BoxSizeX + 1e-9, dx))
    ax.set_yticks(np.arange(0, BoxSizeY + 1e-9, dy))
    ax.grid(True)

    # status_*.bin is written by save_status_bin(): 4 doubles per cell:
    # density, potential, forces_x, forces_y
    raw = np.fromfile(status_path, dtype=np.float64)
    if raw.size % 4 != 0:
        raise ValueError(
            f"Status file {status_path} has {raw.size} float64 values; expected multiple of 4 "
            f"(density, pot, Fx, Fy per cell)."
        )
    grid_status = raw.reshape((-1, 4))
    expected_cells = NgridX * NgridY
    if grid_status.shape[0] != expected_cells:
        raise ValueError(
            f"Status file {status_path} has {grid_status.shape[0]} cells, but params.conf implies "
            f"{expected_cells} (NgridX={NgridX}, NgridY={NgridY})."
        )

    # -----------------------------------------------------------------------
    #               Plot the density as an alpha color grid
    # -----------------------------------------------------------------------
    
    # reshape into the 2D grid; pm.c flattens in row-major [y, x]
    grid_density = grid_status[:, 0].reshape((NgridY, NgridX))  # shape: [y, x]

    # normalize for transparency: lower density -> more transparent
    max_density = grid_density.max() if grid_density.size else 0.0
    alpha_grid = grid_density / (2 * max_density) if max_density > 0 else np.zeros_like(grid_density)

    # plot density on the same grid as particles; edges align to cell boundaries
    x_edges = np.linspace(0, BoxSizeX, NgridX + 1)
    y_edges = np.linspace(0, BoxSizeY, NgridY + 1)
    ax.pcolormesh(
        x_edges,
        y_edges,
        grid_density,
        cmap="viridis",
        shading="auto",
        alpha=alpha_grid,
    )

    # keep cells square for both on-screen and saved figures
    ax.set_xlim(0, BoxSizeX)
    ax.set_ylim(0, BoxSizeY)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    density_out = f"{OUT_DIR}/{method}/density.png"
    print(f"Saving figure to {density_out}")
    fig.savefig(density_out)
    # plt.show()
    plt.close(fig)

    # -----------------------------------------------------------------------
    #               Plot the potential as a color grid
    # -----------------------------------------------------------------------
    
    grid_potential = grid_status[:, 1].reshape((NgridY, NgridX))  # shape: [y, x]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.pcolormesh(
        x_edges,
        y_edges,
        grid_potential,
        cmap="viridis",
        shading="auto",
    )
    ax.set_xlabel("x [kpc]")

    ax.set_xticks(np.arange(0, BoxSizeX + 1e-9, dx))
    ax.set_yticks(np.arange(0, BoxSizeY + 1e-9, dy))
    ax.grid(True)

    fig.tight_layout()
    potential_out = f"{OUT_DIR}/{method}/potential.png"
    print(f"Saving figure to {potential_out}")
    fig.savefig(potential_out)
    # plt.show()
    plt.close(fig)

    # -----------------------------------------------------------------------
    #               Plot the forces as a color grid
    # -----------------------------------------------------------------------
    
    force_components = grid_status[:, 2:4].reshape((NgridY, NgridX, 2))  # shape: [y, x, (Fx, Fy)]
    grid_forces = np.linalg.norm(force_components, axis=2)  # magnitude for visualization
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.pcolormesh(
        x_edges,
        y_edges,
        grid_forces,
        cmap="viridis",
        shading="auto",
    )
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    ax.set_title(f"Particle Mesh - {method}")

    ax.set_xticks(np.arange(0, BoxSizeX + 1e-9, BoxSizeX//NgridX))
    ax.set_yticks(np.arange(0, BoxSizeY + 1e-9, BoxSizeY//NgridY))
    ax.grid(True)

    fig.tight_layout()
    forces_out = f"{OUT_DIR}/{method}/forces.png"
    print(f"Saving figure to {forces_out}")
    fig.savefig(forces_out)
    # plt.show()
    plt.close(fig)

def main():
    params_path = "params.conf"
    
    for method in METHODS:
        positions_path = f"{OUT_DIR}/{method}/positions/positions_0.bin"
        status_path = f"{OUT_DIR}/{method}/status/status_0.bin"
        plot_particles(method, params_path, positions_path, status_path)

if __name__ == "__main__":
    main()