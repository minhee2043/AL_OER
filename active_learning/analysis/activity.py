"""
Generate ternary activity plot for NiCoFe catalyst system.

This script calculates and visualizes the catalytic activity across the
composition space using Boltzmann-weighted kinetic modeling.

Input: Element counts CSV (from sum_element.py)
Output: Ternary activity plot

Usage:
    python activity_plot.py <input_file> [output_file] [E_opt] [T]

"""

import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import sys
import os
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 9,
    'figure.figsize': (3.2, 2.3),
    'figure.dpi': 600,
    'font.weight': 'bold'
})


def generate_composition_grid(steps=18):
    """
    Generate grid of possible ternary compositions.
    """
    step_size = 1.0 / (steps - 1)
    compositions = []
    
    for i in range(steps):
        x = round(i * step_size, 3)
        for j in range(steps - i):
            y = round(j * step_size, 3)
            z = round(1.0 - x - y, 3)
            
            # Ensure composition sums to 1 within numerical precision
            if z >= 0 and abs(x + y + z - 1.0) < 1e-10:
                compositions.append((x, y, z))
    
    return compositions


def calculate_activities(input_file, compositions, E_opt=5.3, T=300):
    """
    Calculate catalytic activity for each composition using microkinetic model.
    
    Activity = Σ_i [probability_i × Boltzmann_factor_i × multiplicity_i]
    
    Args:
        input_file (str): Path to element counts CSV
        compositions (list): List of (Ni, Fe, Co) composition tuples
        E_opt (float): Optimal descriptor value in eV (default: 5.3)
        T (float): Temperature in Kelvin (default: 300)
    """
    kb = 8.617333262e-5  # Boltzmann constant in eV/K
    
    activities = []
    
    print(f"Calculating activities for {len(compositions)} compositions...")
    
    for idx, ratio in enumerate(compositions):
        activity = 0
        
        with open(input_file) as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                try:
                    # Parse row: Ni_count, Fe_count, Co_count, energy_diff, uncertainty, mult
                    Ni_count = int(row[0])
                    Fe_count = int(row[1])
                    Co_count = int(row[2])
                    energy_diff = float(row[3])
                    multiplicity = int(row[5])
                    
                    # Calculate probability for this configuration at given composition
                    # P(config) = (Ni_ratio)^n_Ni × (Fe_ratio)^n_Fe × (Co_ratio)^n_Co
                    probability = (math.pow(ratio[0], Ni_count) * 
                                 math.pow(ratio[1], Fe_count) * 
                                 math.pow(ratio[2], Co_count))
                    
                    # exp(-|ΔG - ΔG_opt| / kT)
                    boltz_factor = math.exp(-abs(energy_diff - E_opt) / (kb * T))
                    
                    # Total contribution weighted by multiplicity
                    contribution = probability * boltz_factor * multiplicity
                    activity += contribution
                    
                except (ValueError, IndexError) as e:
                    continue
        
        activities.append(activity)
        
        if (idx + 1) % 50 == 0:
            print(f"  Calculated {idx + 1}/{len(compositions)} compositions")
    
    return np.array(activities)


def convert_to_ternary_coordinates(compositions):
    y = 0.5 * np.sqrt(3) * compositions[:, 2]  # Co component
    x = compositions[:, 1] + y / np.sqrt(3)    # Fe + Co/2
    
    return x, y


def plot_ternary_activity(compositions, activities, output_file='Activity_ternaryplot.png'):
    """
    Create ternary activity plot with custom color scheme.
    """
    x, y = convert_to_ternary_coordinates(compositions)
    
    colors_custom = ['#3E1F00', '#FF9B66', '#FFFFFF', '#9B8CC5', '#4A3C89']
    custom_cmap = LinearSegmentedColormap.from_list('custom', colors_custom)
    
    fig, ax = plt.subplots()
    
    center_x = (0 + 1 + 0.5) / 3
    center_y = (0 + 0 + 0.5*np.sqrt(3)) / 3
    scale = 1.13
    
    triangle_points = np.array([
        [center_x + (0 - center_x)*scale, center_y + (0 - center_y)*scale],
        [center_x + (1 - center_x)*scale, center_y + (0 - center_y)*scale],
        [center_x + (0.5 - center_x)*scale, center_y + (0.5*np.sqrt(3) - center_y)*scale],
        [center_x + (0 - center_x)*scale, center_y + (0 - center_y)*scale]
    ])
    plt.plot(triangle_points[:,0], triangle_points[:,1], 'k-', linewidth=0.4)
    
    scatter = plt.scatter(x, y, 
                         s=45,                
                         c=activities,        
                         cmap=custom_cmap,    
                         marker='o',
                         edgecolor='none',
                         vmin=0, vmax=activities.max())
    
    text_offset = 0.01
    plt.text(triangle_points[0,0] - text_offset*2, triangle_points[0,1], 'Ni', 
             fontsize=9, ha='right', va='center', weight='bold')
    plt.text(triangle_points[1,0] + text_offset*2, triangle_points[1,1], 'Fe', 
             fontsize=9, ha='left', va='center', weight='bold')
    plt.text(triangle_points[2,0], triangle_points[2,1] + text_offset*1.5, 'Co', 
             fontsize=9, ha='center', va='bottom', weight='bold')
    
    plt.axis('off')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 0.95)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    
    cbar = plt.colorbar(scatter, 
                       cax=cax,
                       ticks=np.linspace(0, activities.max(), 6))
    
    cbar.outline.set_linewidth(0.4)
    cbar.ax.tick_params(width=0.4, length=2, labelsize=8)
    cbar.set_label('Activity', size=8, labelpad=5, weight='bold')
    cbar.ax.set_yticklabels([f'{x:.3f}' for x in np.linspace(0, activities.max(), 6)], 
                           weight='bold')
    
    plt.subplots_adjust(right=0.9)
    plt.savefig(output_file, bbox_inches='tight')
    print(f" Plot saved to: {output_file}")
    
    max_idx = np.argmax(activities)
    optimal_comp = compositions[max_idx]
    print(f"\nOptimal composition (highest activity):")
    print(f"  Ni: {optimal_comp[0]:.3f}")
    print(f"  Fe: {optimal_comp[1]:.3f}")
    print(f"  Co: {optimal_comp[2]:.3f}")
    print(f"  Activity: {activities[max_idx]:.6f}")


def main():
    """
    Main execution function with command-line interface.
    
    Usage:
        python activity_plot.py <input_file> [output_file] [E_opt] [T]
    """
    if len(sys.argv) < 2:
        print("ERROR: Missing required argument\n")
        print("Usage: python activity_plot.py <input_file> [output_file] [E_opt] [T]\n")
        print("Arguments:")
        print("  input_file  : Element counts CSV from sum_element.py")
        print("  output_file : Output plot filename (default: Activity_ternaryplot.png)")
        print("  E_opt       : Optimal descriptor value in eV (default: 5.3)")
        print("  T           : Temperature in Kelvin (default: 300)")
        print("\nExample:")
        print("  python activity_plot.py batch15_count.csv activity.png 5.3 300")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'Activity_ternaryplot.png'
    E_opt = float(sys.argv[3]) if len(sys.argv) > 3 else 5.3
    T = float(sys.argv[4]) if len(sys.argv) > 4 else 300
    
    if not os.path.exists(input_file):
        print(f"ERROR: Input file '{input_file}' not found")
        sys.exit(1)
    
    print(f"Activity calculation parameters:")
    print(f"  Input: {input_file}")
    print(f"  Output: {output_file}")
    print(f"  E_opt: {E_opt} eV")
    print(f"  Temperature: {T} K\n")
    
    try:
        compositions = generate_composition_grid(steps=18)
        compositions = np.array(compositions)
        print(f" Generated {len(compositions)} composition points\n")
        
        activities = calculate_activities(input_file, compositions, E_opt, T)
        print(f" Activity calculation complete\n")
        
        plot_ternary_activity(compositions, activities, output_file)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
