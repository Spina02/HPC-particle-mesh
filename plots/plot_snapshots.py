import argparse
import glob
import os
from io import BytesIO
from multiprocessing import Pool, cpu_count

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "artifacts"

def read_params(params_path="params.conf"):
    """Parse the small params.conf for grid/box sizes."""
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


def list_snapshots(method, kind="status", format="bin"):
    pattern = os.path.join(OUT_DIR, method, kind, f"{kind}_*.{format}")
    files = []
    for path in glob.glob(pattern):
        try:
            idx = int(os.path.splitext(os.path.basename(path))[0].split("/")[-1].split("_")[1])
        except (IndexError, ValueError):
            continue
        files.append((idx, path))
    return [p for _, p in sorted(files, key=lambda x: x[0])]


def load_status(path, NgridY, NgridX, field, format="bin"):
    if format == "txt":
        data = np.loadtxt(path, ndmin=1, dtype=float)
        if data.ndim == 1:
            data = data.reshape(1, -1)
    else:
        data = np.fromfile(path, dtype=np.float64)
        # Binary data is flat; reshape to (N_cells, 4) for the 4 columns
        data = data.reshape(-1, 4)
    
    if field == "density":
        return data[:, 0].reshape((NgridY, NgridX))
    elif field == "potential":
        return data[:, 1].reshape((NgridY, NgridX))
    elif field == "force":
        fx_fy = data[:, 2:4].reshape((NgridY, NgridX, 2))
        return np.sqrt(fx_fy[:, :, 0]**2 + fx_fy[:, :, 1]**2)
    else:
        raise ValueError("field must be one of: density, potential, force")


def maybe_load_positions(path, format="bin"):
    if not os.path.exists(path):
        return None, None
    if format == "txt":
        pts = np.loadtxt(path)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
    else:
        pts = np.fromfile(path, dtype=np.float64)
        # Binary data is flat; reshape to (N_particles, 2) for x, y columns
        pts = pts.reshape(-1, 2)
    return pts[:, 0], pts[:, 1]


def render_single_frame(args):
    """Render a single frame - designed for multiprocessing"""
    path, idx, field, scatter, cmap, vmin, vmax, format, NgridX, NgridY, BoxSizeX, BoxSizeY, positions_template, x_edges, y_edges = args
    
    # Use Agg backend for multiprocessing
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(15, 15))

    if field == "particles":
        x_pts, y_pts = maybe_load_positions(path, format)
        if x_pts is None:
            raise SystemExit(f"Missing positions file matching {path}")
        ax.scatter(x_pts, y_pts, s=1, c="black", alpha=1, linewidths=0, rasterized=True)
    else:
        grid = load_status(path, NgridY, NgridX, field, format)
        x_pts, y_pts = (None, None)
        if scatter:
            x_pts, y_pts = maybe_load_positions(positions_template.format(idx), format)

        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            grid,
            cmap=cmap,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        cbar = fig.colorbar(mesh, ax=ax, shrink=0.8)
        cbar.set_label(field)
        if x_pts is not None and y_pts is not None:
            ax.scatter(x_pts, y_pts, s=2, c="k", alpha=0.5, linewidths=0, rasterized=True)

    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    ax.set_title(f"snapshot {idx}")
    ax.set_xlim(0, BoxSizeX)
    ax.set_ylim(0, BoxSizeY)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    frame = imageio.imread(buf)
    buf.close()
    
    return frame


def render_frames(method, field, scatter, cmap, fps, format):
    NgridX, NgridY, BoxSizeX, BoxSizeY = read_params()
    if field == "particles":
        position_files = list_snapshots(method, "positions", format)
        if not position_files:
            raise SystemExit(f"No snapshots found at {OUT_DIR}/{method}/positions/positions_*.{format}")
        status_files = position_files
    else:
        status_files = list_snapshots(method, "status", format)
        if not status_files:
            raise SystemExit(f"No snapshots found at {OUT_DIR}/{method}/status/status_*.{format}")

    positions_template = os.path.join(OUT_DIR, method, "positions", f"positions_{{}}.{format}")
    x_edges = np.linspace(0, BoxSizeX, NgridX + 1)
    y_edges = np.linspace(0, BoxSizeY, NgridY + 1)

    # Determine a global color scale to avoid flicker (for grid fields)
    vmin = vmax = None
    if field != "particles":
        print("Computing global color scale...")
        grids = []
        for path in status_files[::max(1, len(status_files)//10)]:  # Sample every 10th file for speed
            grid = load_status(path, NgridY, NgridX, field, format)
            grids.append(grid)
        all_grids = np.concatenate([g.flatten() for g in grids])
        vmin = np.percentile(all_grids, 1)  # Use percentiles to avoid outliers
        vmax = np.percentile(all_grids, 99)

    # Prepare arguments for multiprocessing
    frame_args = []
    for path in status_files:
        idx = os.path.splitext(os.path.basename(path))[0].split("/")[-1].split("_")[1]
        frame_args.append((
            path, idx, field, scatter, cmap, vmin, vmax, format,
            NgridX, NgridY, BoxSizeX, BoxSizeY, positions_template, x_edges, y_edges
        ))

    # Use multiprocessing to render frames in parallel
    print(f"Rendering {len(frame_args)} frames using {min(cpu_count(), len(frame_args))} processes...")
    with Pool(processes=min(cpu_count(), len(frame_args))) as pool:
        frames = pool.map(render_single_frame, frame_args)

    duration = 1.0 / fps
    return frames, duration


def main():
    parser = argparse.ArgumentParser(description="Animate PM snapshots.")
    parser.add_argument("--method", default="TSC", help="Method subfolder under artifacts/")
    parser.add_argument(
        "--field",
        default="particles",
        choices=["density", "potential", "force", "particles"],
        help="Field to visualize (use 'particles' for particle-only scatter)",
    )
    parser.add_argument("--out", default=None, help="Output GIF/MP4 path")
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second for GIF")
    parser.add_argument(
        "--scatter",
        action="store_true",
        help="Overlay particle positions if positions_*.txt are present",
    )
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap")
    parser.add_argument("--format", default="bin", choices=["txt", "bin"], help="Format of the data files")
    args = parser.parse_args()

    frames, duration = render_frames(
        method=args.method,
        field=args.field,
        scatter=args.scatter,
        cmap=args.cmap,
        fps=args.fps,
        format=args.format,
    )

    out_path = args.out
    if out_path is None:
        out_path = os.path.join(OUT_DIR, args.method, f"{args.field}.gif")

    print(f"Saving animation to {out_path}...")
    imageio.mimsave(out_path, frames, duration=duration)
    print(f"Saved animation to {out_path}")


if __name__ == "__main__":
    main()