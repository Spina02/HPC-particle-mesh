import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Create results folder if needed
os.makedirs('results', exist_ok=True)

# ----------------------------------
#       PLOT 1: STRONG SCALING
# ----------------------------------
try:
    strong = pd.read_csv('results/strong_scaling.csv')
    strong.columns = strong.columns.str.strip()

    plt.figure(figsize=(10, 6))

    # Sort and compute metrics
    strong = strong.sort_values('cpus_per_task')
    base_time_strong = strong[strong['cpus_per_task'] == 1]['total'].values[0]

    # Speedup = T(1) / T(N), Ideal = N, Efficiency = Speedup / N
    strong['speedup'] = base_time_strong / strong['total']
    strong['ideal'] = strong['cpus_per_task']
    strong['efficiency'] = (strong['speedup'] / strong['cpus_per_task']) * 100

    # Plot speedup (log-log scale)
    plt.plot(strong['cpus_per_task'], strong['speedup'], 'o-', label='Real Speedup', linewidth=2, color='tab:blue')
    plt.plot(strong['cpus_per_task'], strong['ideal'], '--', label='Ideal Speedup', color='grey', alpha=0.7)

    # Labels and style
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup (T1 / Tn)')
    plt.title('Strong Scaling (Fixed Problem Size: 10^6 points)')
    plt.legend(loc='upper left')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    
    # Format axis labels as plain numbers
    ax = plt.gca()
    ax.set_xticks(strong['cpus_per_task'])
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    ax.yaxis.set_major_formatter(plt.ScalarFormatter())

    # Annotate efficiency on data points
    for i, txt in enumerate(strong['efficiency']):
        plt.annotate(f"{txt:.0f}%", 
                    (strong['cpus_per_task'].iloc[i], strong['speedup'].iloc[i]),
                    xytext=(5, -10), textcoords='offset points', fontsize=9, color='red')

    plt.tight_layout()
    plt.savefig('results/strong_scaling.png')
    plt.close()
    
except FileNotFoundError:
    print("File results/strong_scaling.csv not found")

# ----------------------------------
#       PLOT 2: WEAK SCALING
# ----------------------------------
try:
    weak = pd.read_csv('results/weak_scaling.csv')
    weak.columns = weak.columns.str.strip()

    plt.figure(figsize=(10, 6))

    # Sort and compute efficiency
    weak = weak.sort_values('cpus_per_task')
    base_time_weak = weak[weak['cpus_per_task'] == 1]['total'].values[0]

    # Efficiency = T(1) / T(N) - ideal is 100% (constant time)
    weak['efficiency'] = (base_time_weak / weak['total']) * 100

    # Dual axis setup
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # Plot efficiency (left axis)
    line1 = ax1.plot(weak['cpus_per_task'], weak['efficiency'], 'o-', color='tab:green', label='Weak Efficiency', linewidth=2)
    ax1.set_ylabel('Efficiency (%)', color='tab:green', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='tab:green')
    ax1.set_ylim(0, 110)
    ax1.axhline(100, color='grey', linestyle='--', alpha=0.5, label='Ideal Efficiency')

    # Plot execution time (right axis)
    line2 = ax2.plot(weak['cpus_per_task'], weak['total'], 's--', color='tab:red', label='Execution Time', linewidth=1.5, alpha=0.7)
    ax2.set_ylabel('Execution Time (s)', color='tab:red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Common X axis
    ax1.set_xlabel('Number of Threads')
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(weak['cpus_per_task'])
    ax1.xaxis.set_major_formatter(plt.ScalarFormatter())
    ax1.set_title('Weak Scaling (Scaled Problem Size)')
    ax1.grid(True, which="major", ls="-", alpha=0.5)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center left')

    plt.tight_layout()
    plt.savefig('results/weak_scaling.png')
    plt.close()
except FileNotFoundError:
    print("File results/weak_scaling.csv not found")

print("Plots saved to results/")