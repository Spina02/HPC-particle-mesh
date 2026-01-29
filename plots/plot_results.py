import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

RESULTS_DIR = "results-double-no-reorder"

# Create results folder if needed
os.makedirs(RESULTS_DIR, exist_ok=True)

def _strip_cols(df):
    df.columns = df.columns.str.strip()
    return df

# ----------------------------------
#       PLOT 1: STRONG SCALING
# ----------------------------------
try:
    strong = _strip_cols(pd.read_csv(os.path.join(RESULTS_DIR, 'strong_scaling.csv')))

    # fig, ax1 = plt.subplots(figsize=(10, 6))
    fig, ax1 = plt.subplots(figsize=(10, 6))

    strong = strong.sort_values('cpus_per_task')
    base_time_strong = strong[strong['cpus_per_task'] == 1]['total'].values[0]

    strong['speedup'] = base_time_strong / strong['total']
    strong['ideal'] = strong['cpus_per_task']
    strong['efficiency'] = (strong['speedup'] / strong['cpus_per_task']) * 100

    # Left: speedup (log scale)
    line1, = ax1.plot(strong['cpus_per_task'], strong['speedup'], 'o-', label='Real Speedup', linewidth=2, color='tab:blue')
    line2, = ax1.plot(strong['cpus_per_task'], strong['ideal'], '--', label='Ideal Speedup', color='grey', alpha=0.7)
    ax1.set_xlabel('Number of Threads')
    ax1.set_ylabel('Speedup (T1 / Tn)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log', base=2)
    ax1.set_xticks(strong['cpus_per_task'])
    ax1.xaxis.set_major_formatter(plt.ScalarFormatter())
    ax1.yaxis.set_major_formatter(plt.ScalarFormatter())
    ax1.grid(True, which="both", ls="-", alpha=0.5)
    ax1.set_title('Strong Scaling (Fixed Problem Size)')

    for i, txt in enumerate(strong['efficiency']):
        ax1.annotate(f"{txt:.0f}%",
                     (strong['cpus_per_task'].iloc[i], strong['speedup'].iloc[i]),
                     xytext=(5, -10), textcoords='offset points', fontsize=8, color='red')

    # Right: throughput (particles/s)
    lines = [line1, line2]
    if 'throughput' in strong.columns:
        ax2 = ax1.twinx()
        line3, = ax2.plot(strong['cpus_per_task'], strong['throughput'] / 1e6, 's-', color='tab:orange', label='Throughput', linewidth=1.5, alpha=0.9)
        ax2.set_ylabel('Throughput (M particles/s)', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        lines.append(line3)

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'strong_scaling.png'))
    plt.close()

except FileNotFoundError:
    print(f"File {os.path.join(RESULTS_DIR, 'strong_scaling.csv')} not found")

# ----------------------------------
#       PLOT 2: WEAK SCALING
# ----------------------------------
try:
    weak = _strip_cols(pd.read_csv(os.path.join(RESULTS_DIR, 'weak_scaling.csv')))

    plt.figure(figsize=(10, 6))

    weak = weak.sort_values('cpus_per_task')
    base_time_weak = weak[weak['cpus_per_task'] == 1]['total'].values[0]
    weak['efficiency'] = (base_time_weak / weak['total']) * 100

    ax1 = plt.gca()
    ax2 = ax1.twinx()

    line1 = ax1.plot(weak['cpus_per_task'], weak['efficiency'], 'o-', color='tab:green', label='Weak Efficiency', linewidth=2)
    ax1.set_ylabel('Efficiency (%)', color='tab:green', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='tab:green')
    ax1.set_ylim(0, 110)
    ax1.axhline(100, color='grey', linestyle='--', alpha=0.5, label='Ideal Efficiency')

    line2 = ax2.plot(weak['cpus_per_task'], weak['total'], 's--', color='tab:red', label='Execution Time', linewidth=1.5, alpha=0.7)
    ax2.set_ylabel('Execution Time (s)', color='tab:red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='tab:red')

    ax1.set_xlabel('Number of Threads')
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(weak['cpus_per_task'])
    ax1.xaxis.set_major_formatter(plt.ScalarFormatter())
    ax1.set_title('Weak Scaling (Scaled Problem Size)')
    ax1.grid(True, which="major", ls="-", alpha=0.5)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center left')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'weak_scaling.png'))
    plt.close()

except FileNotFoundError:
    print(f"File {os.path.join(RESULTS_DIR, 'weak_scaling.csv')} not found")

# ----------------------------------
#       PLOT 3: GPU SCALING (problem size scaling on one GPU)
# ----------------------------------
try:
    gpu = _strip_cols(pd.read_csv(os.path.join(RESULTS_DIR, 'gpu_scaling.csv')))
    gpu = gpu.sort_values('npoints')

    fig, ax1 = plt.subplots(figsize=(10, 6))

    npoints = gpu['npoints'].values
    labels = [f"{int(g)}" for g in gpu['ngrid_x'].astype(int)]
    x_pos = np.arange(len(npoints))

    # Left: throughput (particles/s) — main metric for GPU scaling
    ax1.plot(x_pos, gpu['throughput'] / 1e6, 'o-', color='tab:blue', linewidth=2, label='Throughput')
    ax1.set_xlabel('Problem size (grid)')
    ax1.set_ylabel('Throughput (M particles/s)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"{int(r['ngrid_x'])}×{int(r['ngrid_y'])}" for _, r in gpu.iterrows()])
    ax1.grid(True, which="major", ls="-", alpha=0.5)
    ax1.set_title('GPU Scaling — Throughput vs Problem Size (single GPU)')

    # Right: total execution time
    ax2 = ax1.twinx()
    ax2.plot(x_pos, gpu['total'], 's--', color='tab:red', linewidth=1.5, alpha=0.8, label='Total time')
    ax2.set_ylabel('Execution time (s)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    ax1.legend(loc='upper right')
    ax2.legend(loc='center right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'gpu_scaling.png'))
    plt.close()

    # Optional: time breakdown (stacked or bar)
    fig, ax = plt.subplots(figsize=(10, 6))
    cols = ['density', 'fft', 'potential', 'forces', 'interpolation', 'kick', 'drift']
    if all(c in gpu.columns for c in cols):
        bottom = np.zeros(len(gpu))
        colors = plt.cm.tab10(np.linspace(0, 1, len(cols)))
        for i, c in enumerate(cols):
            ax.bar(x_pos, gpu[c], bottom=bottom, label=c, color=colors[i])
            bottom = bottom + gpu[c].values
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{int(r['ngrid_x'])}×{int(r['ngrid_y'])}" for _, r in gpu.iterrows()])
        ax.set_xlabel('Problem size (grid)')
        ax.set_ylabel('Time (s)')
        ax.set_title('GPU Scaling — Time breakdown by phase')
        ax.legend(loc='upper left', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'gpu_scaling_breakdown.png'))
        plt.close()

except FileNotFoundError:
    print(f"File {os.path.join(RESULTS_DIR, 'gpu_scaling.csv')} not found")

# ----------------------------------
#       PLOT 4: BACKEND COMPARISON (serial, vec, OMP best, GPU best)
# ----------------------------------
def _load_one_row(path):
    try:
        df = _strip_cols(pd.read_csv(path))
        if len(df) == 0 or 'total' not in df.columns:
            return None
        return df.iloc[0]
    except FileNotFoundError:
        return None

comparison = []  # list of (label, total_time, throughput)

# Serial: single run
row = _load_one_row(os.path.join(RESULTS_DIR, 'serial_timing.csv'))
if row is not None:
    comparison.append(('Serial', float(row['total']), float(row['throughput'])))

# Vec: single run
row = _load_one_row(os.path.join(RESULTS_DIR, 'vec_timing.csv'))
if row is not None:
    comparison.append(('Vec', float(row['total']), float(row['throughput'])))

# OMP best: row with max throughput in strong_scaling
try:
    strong = _strip_cols(pd.read_csv(os.path.join(RESULTS_DIR, 'strong_scaling.csv')))
    if 'throughput' in strong.columns and len(strong) > 0:
        best = strong.loc[strong['throughput'].idxmax()]
        comparison.append(('OMP (best)', float(best['total']), float(best['throughput'])))
except FileNotFoundError:
    pass

# GPU best: row with max throughput in gpu_scaling
try:
    gpu = _strip_cols(pd.read_csv(os.path.join(RESULTS_DIR, 'gpu_scaling.csv')))
    if len(gpu) > 0:
        best = gpu.loc[gpu['throughput'].idxmax()]
        comparison.append(('GPU (best)', float(best['total']), float(best['throughput'])))
except FileNotFoundError:
    pass

if len(comparison) >= 1:
    labels = [x[0] for x in comparison]
    times = np.array([x[1] for x in comparison])
    throughputs = np.array([x[2] for x in comparison]) / 1e6  # M particles/s

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(labels))
    width = 0.6

    # Total time (s)
    bars1 = ax1.bar(x, times, width, align='center', color=['tab:blue', 'tab:green', 'tab:orange', 'tab:red'][:len(labels)])
    ax1.set_ylabel('Total time (s)')
    ax1.set_xlabel('Backend')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title('Execution time comparison')
    ax1.bar_label(bars1, fmt='%.2f', padding=2)

    # Throughput (M particles/s)
    bars2 = ax2.bar(x, throughputs, width, align='center', color=['tab:blue', 'tab:green', 'tab:orange', 'tab:red'][:len(labels)])
    ax2.set_ylabel('Throughput (M particles/s)')
    ax2.set_xlabel('Backend')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_title('Throughput comparison')
    ax2.bar_label(bars2, fmt='%.1f', padding=2)

    plt.suptitle('Backend comparison (Serial, Vec, OMP best-case, GPU best-case)', fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'backend_comparison.png'))
    plt.close()
else:
    print("No data found for backend comparison (need at least one of serial_timing, vec_timing, strong_scaling, gpu_scaling)")

# ----------------------------------
#       PLOT 5: SERIAL vs VEC — TIME BREAKDOWN BY FUNCTION
# ----------------------------------
phase_cols = ['density', 'fft', 'potential', 'forces', 'interpolation', 'kick', 'drift', 'reorder']

def _load_breakdown(path):
    try:
        df = _strip_cols(pd.read_csv(path))
        if len(df) == 0:
            return None
        row = df.iloc[0]
        if not all(c in row.index for c in phase_cols):
            return None
        return {c: float(row[c]) for c in phase_cols if c in row.index}
    except FileNotFoundError:
        return None

serial_breakdown = _load_breakdown(os.path.join(RESULTS_DIR, 'serial_timing.csv'))
vec_breakdown = _load_breakdown(os.path.join(RESULTS_DIR, 'vec_timing.csv'))

if serial_breakdown is not None and vec_breakdown is not None:
    # Use only columns present in both
    cols = [c for c in phase_cols if c in serial_breakdown and c in vec_breakdown]
    if cols:
        x = np.arange(len(cols))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        serial_times = [serial_breakdown[c] for c in cols]
        vec_times = [vec_breakdown[c] for c in cols]
        
        bars_serial = ax.bar(x - width / 2, serial_times, width, label='Serial', color='tab:blue')
        bars_vec = ax.bar(x + width / 2, vec_times, width, label='Vec', color='tab:green')

        # Add speedup percentage labels over the bars
        for i, (s_time, v_time) in enumerate(zip(serial_times, vec_times)):
            if v_time > 0:
                speedup = s_time / v_time
                max_height = max(s_time, v_time)
                ax.annotate(f'{speedup:.2f}x',
                           xy=(x[i], max_height),
                           xytext=(0, 5),
                           textcoords='offset points',
                           ha='center', va='bottom',
                           fontsize=9, fontweight='bold', color='red')

        ax.set_ylabel('Time (s)')
        ax.set_xlabel('Phase')
        ax.set_xticks(x)
        ax.set_xticklabels(cols, rotation=0, ha='center')
        ax.set_title('Serial vs Vec — Time breakdown by function')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'serial_vs_vec.png'))
        plt.close()
        print(f"Saved {os.path.join(RESULTS_DIR, 'serial_vs_vec.png')}")
else:
    if serial_breakdown is None:
        print(f"{os.path.join(RESULTS_DIR, 'serial_timing.csv')} not found or missing phase columns for breakdown")
    if vec_breakdown is None:
        print(f"{os.path.join(RESULTS_DIR, 'vec_timing.csv')} not found or missing phase columns for breakdown")

print(f"Plots saved to {RESULTS_DIR}/")