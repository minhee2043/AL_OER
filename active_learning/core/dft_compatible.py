"""
Generate all possible DFT-compatible surface configurations for 2×2×4 slab.

This script generates 6561 (3^8) possible configurations where each of the 8
positions in a 2×2×4 slab structure can be occupied by Ni, Co, or Fe.

The 8 positions correspond to specific locations in the surface structure,
and each configuration is assigned a multiplicity based on the degeneracy of
equivalent atomic arrangements.

Output Files:
    - possibleFp.csv: Feature vectors (15 features + multiplicity)
    - index_metal.csv: Metal sequences for each configuration
"""

import numpy as np
from itertools import product
from helperMethods import multiplicity
import csv


def generate_surface_configurations():
    """
    Generate all possible surface configurations and their feature vectors.
    """
    # Generate all 6561 possible surface configurations (3 metals, 8 positions)
    possible_surface = list(product(['Ni', 'Co', 'Fe'], repeat=8))
    
    n_configs = len(possible_surface)
    mults = np.zeros(n_configs)
    features = np.zeros((n_configs, 15))  # 5 zones × 3 metals = 15 features
    
    print(f"Generating {n_configs} surface configurations...")
    
    for i, config in enumerate(possible_surface):
        # Initialize metal counters for each coordination zone
        # Zone definitions based on geometric positions around adsorption site
        ensemble = {'Ni': 0, 'Co': 0, 'Fe': 0}          # Atoms forming ads site
        surface_near = {'Ni': 0, 'Co': 0, 'Fe': 0}      # Nearest surface neighbors
        subsurface_near = {'Ni': 0, 'Co': 0, 'Fe': 0}   # Nearest subsurface
        surface_far = {'Ni': 0, 'Co': 0, 'Fe': 0}       # Farther surface neighbors
        subsurface_far = {'Ni': 0, 'Co': 0, 'Fe': 0}    # Farther subsurface
        
        # Map positions to coordination zones
        # Positions 0-7 correspond to specific locations in the 2×2×4 slab
        subsurface_far[config[0]] += 1
        subsurface_far[config[1]] += 1
        subsurface_far[config[2]] += 1
        
        surface_far[config[2]] += 1
        surface_far[config[3]] += 2
        
        ensemble[config[4]] += 1
        surface_near[config[4]] += 2
        
        ensemble[config[5]] += 1
        surface_near[config[5]] += 2
        
        ensemble[config[6]] += 1
        subsurface_near[config[6]] += 1
        
        surface_near[config[7]] += 2
        subsurface_near[config[7]] += 2
        
        ensemble_vals = list(ensemble.values())
        surface_near_vals = list(surface_near.values())
        subsurface_near_vals = list(subsurface_near.values())
        surface_far_vals = list(surface_far.values())
        subsurface_far_vals = list(subsurface_far.values())
        
        # Calculate multiplicity for each zone
        ensemble_mult = multiplicity(2, [x for x in ensemble_vals if x != 0])
        surface_near_mult = multiplicity(4, [x for x in surface_near_vals if x != 0])
        subsurface_near_mult = multiplicity(2, [x for x in subsurface_near_vals if x != 0])
        surface_far_mult = multiplicity(2, [x for x in surface_far_vals if x != 0])
        subsurface_far_mult = multiplicity(1, [x for x in subsurface_far_vals if x != 0])
        
        total_mult = (ensemble_mult * surface_near_mult * subsurface_near_mult * 
                      surface_far_mult * subsurface_far_mult)
        
        features[i] = np.array(
            ensemble_vals + surface_near_vals + subsurface_near_vals + 
            surface_far_vals + subsurface_far_vals
        )
        mults[i] = total_mult

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{n_configs} configurations")
    
    return features, mults, possible_surface


def save_outputs(features, multiplicities, configurations, 
                feature_file='possibleFp.csv', 
                index_file='index_metal.csv'):
    """
    Save generated configurations to CSV files.
    
    Args:
        features (np.ndarray): Feature vectors
        multiplicities (np.ndarray): Multiplicity values
        configurations (list): Metal sequences
        feature_file (str): Output file for features
        index_file (str): Output file for metal indices
    """
    with open(index_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(configurations)
    
    output = np.c_[features, multiplicities]
    
    np.savetxt(feature_file, output, fmt=['%d']*16, delimiter=',')
    
    print(f"\n Saved {len(configurations)} configurations")
    print(f" Features saved to: {feature_file}")
    print(f" Metal indices saved to: {index_file}")


def main():
    import sys
    
    feature_file = sys.argv[1] if len(sys.argv) > 1 else 'possibleFp.csv'
    index_file = sys.argv[2] if len(sys.argv) > 2 else 'index_metal.csv'
    
    features, multiplicities, configurations = generate_surface_configurations()
    
    save_outputs(features, multiplicities, configurations, feature_file, index_file)
    
    print(f"\nSummary:")
    print(f"  Total configurations: {len(configurations)}")
    print(f"  Feature dimensions: {features.shape}")
    print(f"  Multiplicity range: {multiplicities.min():.0f} - {multiplicities.max():.0f}")


if __name__ == "__main__":
    main()
