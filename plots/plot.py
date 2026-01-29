"""
Plot density, potential, or force magnitude for a specified PM iteration.
Also supports particles-only (no fields).

Usage:
  python plot.py --backend hpc --iteration 0 --field all
  python plot.py --backend hpc --iteration 12 --field density --field potential --scatter
  python plot.py --backend hpc --iteration 0 --field particles
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "artifacts"
FIELDS = ("density", "potential", "force")
FIELD_CHOICES = (*FIELDS, "particles", "all")


def read_params(params_path="params.conf"):
    """Parse params.conf for grid/box sizes (same logic as plot_snapshots)."""
    cfg = {}
    with open(params_path, "r") as fp:
        for line in fp:
            if "=" not in line or line.strip().startswith("#"):
                continue
            key, val = line.split("=", 1)
            cfg[key.strip()] = val.split("#")[0].strip()
    NgridX = int(cfg["NgridX"])
    NgridY = int(cfg["NgridY"])
    BoxSizeX = float(cfg["BoxSizeX"])
    BoxSizeY = float(cfg["BoxSizeY"])
    return NgridX, NgridY, BoxSizeX, BoxSizeY


def load_status(path, NgridY, NgridX, field):
    """Load status_*.bin and return 2D grid for the requested field."""
    data = np.fromfile(path, dtype=np.float64)
    n_cells = data.size // 4
    if data.size != 4 * n_cells:
        raise ValueError(
            f"Status file {path} has {data.size} float64 values; "
            "expected multiple of 4 (density, potential, Fx, Fy per cell)."
        )
    data = data.reshape(-1, 4)
    if field == "density":
        return data[:, 0].reshape((NgridY, NgridX))
    if field == "potential":
        return data[:, 1].reshape((NgridY, NgridX))
    if field == "force":
        fx_fy = data[:, 2:4].reshape((NgridY, NgridX, 2))
        return np.sqrt(fx_fy[:, :, 0] ** 2 + fx_fy[:, :, 1] ** 2)
    raise ValueError("field must be one of: density, potential, force")


def _grid_from_status(path, NgridX, NgridY):
    """Return (NgridX, NgridY) to use; infer from file if params mismatch."""
    raw = np.fromfile(path, dtype=np.float64)
    n_cells = raw.size // 4
    if raw.size != 4 * n_cells:
        raise ValueError(
            f"Status file {path} has {raw.size} float64 values; "
            "expected multiple of 4."
        )
    expected = NgridX * NgridY
    if n_cells == expected:
        return NgridX, NgridY
    s = int(n_cells ** 0.5)
    if s * s != n_cells:
        raise ValueError(
            f"Status file {path} has {n_cells} cells; params expect "
            f"NgridX*NgridY={expected}. Cannot infer non-square grid."
        )
    return s, s


def maybe_load_positions(path):
    """Load positions_*.bin if present; return (x, y) or (None, None)."""
    if not os.path.exists(path):
        return None, None
    pts = np.fromfile(path, dtype=np.float64)
    pts = pts.reshape(-1, 2)
    return pts[:, 0], pts[:, 1]


def plot_field(
    *,
    backend,
    iteration,
    field,
    scatter,
    out_dir,
    params_path,
):
    NgridX, NgridY, BoxSizeX, BoxSizeY = read_params(params_path)
    status_path = os.path.join(OUT_DIR, backend, "status", f"status_{iteration}.bin")
    positions_path = os.path.join(
        OUT_DIR, backend, "positions", f"positions_{iteration}.bin"
    )

    if field == "particles":
        if os.path.exists(status_path):
            NgridX, NgridY = _grid_from_status(status_path, NgridX, NgridY)
        x_pts, y_pts = maybe_load_positions(positions_path)
        if x_pts is None or y_pts is None:
            raise FileNotFoundError(f"Positions file not found: {positions_path}")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(x_pts, y_pts, s=40, c="black", alpha=1, linewidths=0, rasterized=True)
        ax.set_xlabel("x [kpc]")
        ax.set_ylabel("y [kpc]")
        # ax.set_title(f"Particles — {backend} (iteration {iteration})")
        iteration = 1
        ax.set_title(f"Particles (iteration {iteration})")
        ax.set_xlim(0, BoxSizeX)
        ax.set_ylim(0, BoxSizeY)
        ax.set_aspect("equal", adjustable="box")
        dx = BoxSizeX / NgridX
        dy = BoxSizeY / NgridY
        step_x = max(1, NgridX // 20)
        step_y = max(1, NgridY // 20)
        ax.set_xticks(np.arange(0, BoxSizeX + 1e-9, dx * step_x))
        ax.set_yticks(np.arange(0, BoxSizeY + 1e-9, dy * step_y))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"particles_iter{iteration}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")
        return

    if not os.path.exists(status_path):
        raise FileNotFoundError(f"Status file not found: {status_path}")

    NgridX, NgridY = _grid_from_status(status_path, NgridX, NgridY)
    grid = load_status(status_path, NgridY, NgridX, field)
    x_edges = np.linspace(0, BoxSizeX, NgridX + 1)
    y_edges = np.linspace(0, BoxSizeY, NgridY + 1)

    x_pts, y_pts = (None, None)
    if scatter:
        x_pts, y_pts = maybe_load_positions(positions_path)

    import matplotlib.colors as mcolors

    base = plt.get_cmap("GnBu")
    trunc = mcolors.LinearSegmentedColormap.from_list(
        "GnBu",
        base(np.linspace(0.5, 0.9, 256))
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        grid,
        cmap=trunc,
        # cmap="viridis",
        shading="auto",
    )
    # cbar = fig.colorbar(mesh, ax=ax, shrink=0.8)
    label = "Force magnitude" if field == "force" else field.capitalize()
    # cbar.set_label(label)

    if x_pts is not None and y_pts is not None:
        ax.scatter(x_pts, y_pts, s=40, c="k", alpha=0.5, linewidths=0, rasterized=True)

    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    # ax.set_title(f"{label} — {backend} (iteration {iteration})")
    ax.set_title(f"{label} (iteration {iteration})")
    ax.set_xlim(0, BoxSizeX)
    ax.set_ylim(0, BoxSizeY)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{field}_iter{iteration}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot density, potential, or force magnitude for a specified PM iteration.",
    )
    parser.add_argument(
        "--backend",
        default="hpc",
        choices=["serial", "vec", "hpc", "gpu"],
        help="Backend subfolder under artifacts/",
    )
    parser.add_argument(
        "--iteration",
        "--iter",
        type=int,
        default=0,
        dest="iteration",
        help="Iteration index (e.g. status_0.bin, positions_0.bin)",
    )
    parser.add_argument(
        "--field",
        choices=list(FIELD_CHOICES),
        default=[],
        action="append",
        help="Field to plot (can repeat). 'particles' = only particles, no fields. Omit for all fields.",
    )
    parser.add_argument(
        "--scatter",
        action="store_true",
        help="Overlay particle positions if positions_<iteration>.bin exists",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory for PNGs (default: artifacts/<backend>)",
    )
    parser.add_argument(
        "--params",
        default="params.conf",
        help="Path to params.conf",
    )
    args = parser.parse_args()

    out_dir = args.out
    if out_dir is None:
        out_dir = os.path.join(OUT_DIR, args.backend)

    raw_fields = args.field
    if not raw_fields or "all" in raw_fields:
        fields = list(FIELDS)  # all = density, potential, force
        if "particles" in raw_fields:
            fields.append("particles")  # keep explicitly requested particles
    else:
        seen = set()
        unique = []
        for f in raw_fields:
            if f not in seen:
                seen.add(f)
                unique.append(f)
        fields = unique

    for field in fields:
        try:
            plot_field(
                backend=args.backend,
                iteration=args.iteration,
                field=field,
                scatter=args.scatter,
                out_dir=out_dir,
                params_path=args.params,
            )
        except FileNotFoundError as e:
            print(e)


if __name__ == "__main__":
    main()
