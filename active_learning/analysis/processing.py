"""
Calculate element counts from GPR predictions for activity calculation.

This script processes GPR prediction outputs to calculate the total count
of each metal element (Ni, Fe, Co) in surface configurations, which is
needed for the activity calculation step.

Input: GPR predictions CSV (from mygaussian.py)
Output: Element counts CSV with format: Ni_count, Fe_count, Co_count, 
        energy_diff, uncertainty, multiplicity

Usage:
    python sum_element.py <input_file> [output_file]
    
Example:
    python sum_element.py GPR_batch15.csv batch15_count.csv
"""

import numpy as np
import pandas as pd
import sys
import os


def calculate_element_counts(input_file, max_rows=280000):
    """
    Calculate element counts for each configuration from GPR predictions.
    
    This function sums metal counts across all coordination zones to determine
    the total composition of each surface configuration.
    
    Args:
        input_file (str): Path to GPR predictions CSV file
                         Expected format: 15 features + multiplicity + predictions
        max_rows (int): Maximum number of rows to process (default: 280000)
        
    Returns:
        np.ndarray: Processed data with shape (n_configs, 6)
                   Columns: [Ni_count, Fe_count, Co_count, energy_diff, 
                            uncertainty, multiplicity]
    
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If input file has incorrect format
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Initialize arrays
    Ni = np.zeros(max_rows)
    Fe = np.zeros(max_rows)
    Co = np.zeros(max_rows)
    diffs = np.zeros(max_rows)
    uncertain = np.zeros(max_rows)
    multiplicity = np.zeros(max_rows)
    
    print(f"Processing: {input_file}")
    
    with open(input_file, 'r') as handle:
        for i, line in enumerate(handle):
            # Parse line into elements
            elements = line.strip().split(',')
            
            # Validate line format
            if len(elements) < 22:
                print(f"Warning: Line {i+1} has only {len(elements)} columns, expected 22+")
                continue
            
            try:
                # Extract features (f1-f15) and predictions
                features = [int(x) for x in elements[:15]]
                mult = int(elements[15])
                diff = float(elements[18])  # E_O - E_OH
                uncertainty = float(elements[21])
                
                # Calculate element counts by summing every third feature
                # Features are organized as: Ni1,Fe1,Co1, Ni2,Fe2,Co2, ...
                Ni[i] = sum(features[j] for j in range(0, 15, 3))
                Fe[i] = sum(features[j] for j in range(1, 15, 3))
                Co[i] = sum(features[j] for j in range(2, 15, 3))
                
                # Store other values
                diffs[i] = diff
                uncertain[i] = uncertainty
                multiplicity[i] = mult
                
            except (ValueError, IndexError) as e:
                print(f"Warning: Error parsing line {i+1}: {e}")
                continue
            
            # Progress indicator
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1} lines...")
            
            if i >= max_rows - 1:
                break
    
    # Trim arrays to actual size
    actual_size = i + 1
    Ni = Ni[:actual_size]
    Fe = Fe[:actual_size]
    Co = Co[:actual_size]
    diffs = diffs[:actual_size]
    uncertain = uncertain[:actual_size]
    multiplicity = multiplicity[:actual_size]
    
    # Combine all data
    output = np.column_stack([Ni, Fe, Co, diffs, uncertain, multiplicity])
    
    return output


def save_results(output_data, output_file):
    """
    Save processed results to CSV file.
    
    Args:
        output_data (np.ndarray): Processed data to save
        output_file (str): Path to output CSV file
    """
    np.savetxt(
        output_file,
        output_data,
        fmt=['%d', '%d', '%d', '%.5f', '%.5f', '%d'],
        delimiter=','
    )
    print(f"✓ Results saved to: {output_file}")


def main():
    """
    Main execution function with command-line interface.
    
    Usage:
        python sum_element.py <input_file> [output_file]
    """
    # Check command line arguments
    if len(sys.argv) < 2:
        print("ERROR: Missing required argument\n")
        print("Usage: python sum_element.py <input_file> [output_file]\n")
        print("Arguments:")
        print("  input_file  : GPR predictions CSV from mygaussian.py")
        print("  output_file : Output CSV with element counts (optional)")
        print("\nExample:")
        print("  python sum_element.py GPR_batch15.csv batch15_count.csv")
        print("\nThe input file should have format:")
        print("  15 features, multiplicity, predictions, uncertainties")
        sys.exit(1)
    
    # Parse arguments
    input_file = sys.argv[1]
    
    # Default output filename based on input
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        # Generate default output name from input name
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_count.csv"
    
    try:
        # Process the data
        output_data = calculate_element_counts(input_file)
        
        # Save results
        save_results(output_data, output_file)
        
        # Print summary
        print(f"\n✓ Processed {len(output_data)} configurations")
        print(f"\nElement count ranges:")
        print(f"  Ni: {output_data[:,0].min():.0f} - {output_data[:,0].max():.0f}")
        print(f"  Fe: {output_data[:,1].min():.0f} - {output_data[:,1].max():.0f}")
        print(f"  Co: {output_data[:,2].min():.0f} - {output_data[:,2].max():.0f}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
