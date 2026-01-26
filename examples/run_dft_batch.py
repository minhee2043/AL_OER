"""
DFT Calculation Workflow for Active Learning Batch

This script automates the process of:
1. Reading GPR-suggested surface configurations from batch{N}_metal.csv
2. Building slab structures with correct lattice parameters
3. Running DFT calculations (clean slab + O adsorption)
4. Extracting features and energies
5. Creating training data for next iteration

Usage:
    python run_dft_batch.py --batch 15 --start 16 --end 20
    
Requirements:
    - ASE with VASP calculator
    - batch{N}_metal.csv from active learning suggestions
    - Configured VASP setup
"""

import os
import sys
import argparse
import numpy as np
import copy
from ase import Atoms
from ase.build import fcc111
from ase.calculators.vasp import Vasp2
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.build import add_adsorbate
from ase.io import read, write
import ase.db

# Import from active_learning package
from active_learning.core.features import Slab


class DFTBatchRunner:
    """
    Automate DFT calculations for active learning batches.
    
    Example:
        >>> runner = DFTBatchRunner(batch_number=15)
        >>> runner.read_suggestions('batch15_metal.csv', start_idx=16, end_idx=20)
        >>> runner.run_all_calculations()
        >>> runner.save_training_data('DFT_O_batch15.csv')
    """
    
    def __init__(self, batch_number, metal_types=['Ni', 'Fe', 'Co'], 
                 lattice_params=None, zones=None):
        """
        Initialize DFT batch runner.
        
        Args:
            batch_number (int): Current batch number
            metal_types (list): Metal types in order (default: ['Ni', 'Fe', 'Co'])
            lattice_params (dict): Lattice parameters for each metal (Å)
            zones (list): Feature zones to extract
        """
        self.batch_number = batch_number
        self.metal_types = metal_types
        
        # Default lattice parameters (Å)
        if lattice_params is None:
            self.lattice_params = {'Ni': 3.53, 'Fe': 3.571, 'Co': 3.486}
        else:
            self.lattice_params = lattice_params
        
        # Default feature zones
        if zones is None:
            self.zones = ['ens', 'sn', 'ssn', 'sf', 'ssf']
        else:
            self.zones = zones
        
        # Storage
        self.configurations = []
        self.results = []
        
    def read_suggestions(self, metal_file, start_idx=0, end_idx=None):
        """
        Read suggested configurations from batch{N}_metal.csv.
        
        Args:
            metal_file (str): Path to metal configuration file
            start_idx (int): Starting index in file (0-indexed)
            end_idx (int): Ending index (exclusive), None = read all
        """
        print(f"\n{'='*60}")
        print(f"Reading suggestions from: {metal_file}")
        print(f"{'='*60}")
        
        configs = []
        with open(metal_file, 'r') as f:
            lines = f.readlines()
            
            if end_idx is None:
                end_idx = len(lines)
            
            for i, line in enumerate(lines[start_idx:end_idx], start=start_idx):
                metals = line.strip().split(',')[:8]
                configs.append({
                    'index': i,
                    'metals': metals,
                    'composition': self._get_composition(metals)
                })
        
        self.configurations = configs
        print(f"✓ Loaded {len(configs)} configurations (indices {start_idx}-{end_idx-1})")
        
        for config in configs[:3]:  # Show first 3
            print(f"  Config {config['index']}: {config['composition']}")
        if len(configs) > 3:
            print(f"  ... and {len(configs)-3} more")
        
        return configs
    
    def _get_composition(self, metal_list):
        """Get composition summary from metal list."""
        from collections import Counter
        counts = Counter(metal_list)
        return ', '.join([f"{metal}:{count*2}" for metal, count in sorted(counts.items())])
    
    def build_slab(self, metal_list, config_index):
        """
        Build slab structure from metal list.
        
        Args:
            metal_list (list): 8 metal symbols for the 2x2x4 slab
            config_index (int): Configuration index for labeling
        
        Returns:
            ase.Atoms: Slab structure ready for calculation
        """
        # Calculate weighted average lattice parameter
        metal_counts = {metal: metal_list.count(metal) * 2 for metal in self.metal_types}
        mixing_norm = sum(metal_counts.values())
        
        avg_lattice = sum(
            metal_counts[metal] * self.lattice_params[metal] 
            for metal in self.metal_types
        ) / mixing_norm
        
        print(f"\n  Building slab {config_index}:")
        print(f"    Composition: {metal_counts}")
        print(f"    Lattice parameter: {avg_lattice:.4f} Å")
        
        slab = fcc111('Ni', (2, 2, 4), a=avg_lattice, vacuum=8, orthogonal=True)
        slab.pbc = (True, True, True)
        
        for i in range(8):
            slab[i].symbol = metal_list[i]
            slab[i+8].symbol = metal_list[i]  # Periodic copy
        
        constraint = FixAtoms(indices=[atom.index for atom in slab if atom.tag in [3, 4]])
        slab.set_constraint(constraint)
        
        return slab
    
    def run_slab_calculation(self, slab, config_index, fmax=0.05):
        """
        Run DFT calculation for clean slab.
        
        Args:
            slab (ase.Atoms): Slab structure
            config_index (int): Configuration index
            fmax (float): Force convergence criterion (eV/Å)
        
        Returns:
            tuple: (optimized_slab, energy)
        """
        # Setup VASP calculator
        calc = Vasp2(
            directory=f'vaspcalc_b{self.batch_number}/ncf_{config_index}',
            lreal='False',
            xc='rpbe',
            encut=450,
            ediff=1e-5,
            algo='normal',
            gga='rp',
            prec='accurate',
            isym=2,
            ismear=0,
            sigma=0.05,
            ispin=1,
            ibrion=2,
            isif=2,
            kspacing=0.3,
            lorbit=10,
            lwave=False,
            npar=8,
            istart=0,
            icharg=2,
            nelm=120
        )
        
        slab.calc = calc
        
        print(f"  Running clean slab optimization...")
        traj_file = f'traj_b{self.batch_number}/ncf_{config_index}.traj'
        os.makedirs(os.path.dirname(traj_file), exist_ok=True)
        
        opt = BFGS(slab, trajectory=traj_file)
        opt.run(fmax=fmax)
        
        energy = slab.get_potential_energy()
        print(f"  ✓ Clean slab energy: {energy:.4f} eV")
        
        return slab, energy
    
    def run_adsorbate_calculation(self, slab, config_index, 
                                  adsorbate='O', site='fcc', height=2.0, fmax=0.05):
        """
        Run DFT calculation for slab with adsorbate.
        
        Args:
            slab (ase.Atoms): Optimized clean slab
            config_index (int): Configuration index
            adsorbate (str): Adsorbate species (default: 'O')
            site (str): Adsorption site (default: 'fcc')
            height (float): Initial adsorbate height (Å)
            fmax (float): Force convergence criterion (eV/Å)
        
        Returns:
            tuple: (optimized_slab_ads, energy)
        """
        slab_ads = copy.deepcopy(slab)
        add_adsorbate(slab_ads, adsorbate, height=height, position=site)
        
        # Setup VASP calculator
        calc = Vasp2(
            directory=f'vaspcalc_b{self.batch_number}/ncf_{config_index}{adsorbate}',
            lreal='False',
            xc='rpbe',
            encut=450,
            ediff=1e-5,
            algo='normal',
            gga='rp',
            prec='accurate',
            isym=2,
            ismear=0,
            sigma=0.05,
            ispin=1,
            ibrion=2,
            isif=2,
            kspacing=0.3,
            lorbit=10,
            lwave=False,
            npar=8,
            istart=0,
            icharg=2,
            nelm=120
        )
        
        slab_ads.calc = calc
        
        print(f"  Running {adsorbate} adsorption optimization...")
        traj_file = f'traj_b{self.batch_number}/ncf_{config_index}{adsorbate}.traj'
        
        opt = BFGS(slab_ads, trajectory=traj_file)
        opt.run(fmax=fmax)
        
        energy = slab_ads.get_potential_energy()
        print(f"  ✓ {adsorbate} adsorption energy: {energy:.4f} eV")
        
        return slab_ads, energy
    
    def extract_features(self, slab_ads):
        """
        Extract features from optimized adsorbate structure.
        
        Args:
            slab_ads (ase.Atoms): Optimized slab with adsorbate
        
        Returns:
            np.ndarray: Feature vector
        """
        slab_obj = Slab(slab_ads)
        features = np.array(slab_obj.features(
            metals=self.metal_types,
            onTop=False,
            zones=self.zones
        ))
        return features
    
    def run_single_calculation(self, config_index, metal_list, 
                              adsorbate='O', reference_energy=3.489):
        """
        Run complete workflow for single configuration.
        
        Args:
            config_index (int): Configuration index
            metal_list (list): 8 metal symbols
            adsorbate (str): Adsorbate species
            reference_energy (float): Reference energy for adsorption (eV)
        
        Returns:
            dict: Results containing features and energies
        """
        print(f"\n{'='*60}")
        print(f"Configuration {config_index}")
        print(f"{'='*60}")
        
        try:
            # Build slab
            slab = self.build_slab(metal_list, config_index)
            
            # Run clean slab calculation
            slab_opt, e_slab = self.run_slab_calculation(slab, config_index)
            
            # Run adsorbate calculation
            slab_ads, e_ads = self.run_adsorbate_calculation(
                slab_opt, config_index, adsorbate=adsorbate
            )
            
            # Extract features
            features = self.extract_features(slab_ads)
            
            # Calculate adsorption energy
            e_adsorption = e_ads - e_slab + reference_energy
            
            result = {
                'index': config_index,
                'features': features,
                'e_slab': e_slab,
                'e_ads': e_ads,
                'e_adsorption': e_adsorption,
                'success': True
            }
            
            print(f"\n  ✓ SUCCESS")
            print(f"    Features: {features}")
            print(f"    E_slab: {e_slab:.4f} eV")
            print(f"    E_ads: {e_ads:.4f} eV")
            print(f"    E_adsorption: {e_adsorption:.4f} eV")
            
            return result
            
        except Exception as e:
            print(f"\n  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'index': config_index,
                'success': False,
                'error': str(e)
            }
    
    def run_all_calculations(self, adsorbate='O'):
        """
        Run calculations for all loaded configurations.
        
        Args:
            adsorbate (str): Adsorbate species (default: 'O')
        """
        print(f"\n{'='*60}")
        print(f"Starting batch {self.batch_number} calculations")
        print(f"Total configurations: {len(self.configurations)}")
        print(f"{'='*60}")
        
        results = []
        for config in self.configurations:
            result = self.run_single_calculation(
                config['index'],
                config['metals'],
                adsorbate=adsorbate
            )
            results.append(result)
        
        self.results = results
        
        successful = sum(1 for r in results if r['success'])
        print(f"\n{'='*60}")
        print(f"Batch {self.batch_number} Complete")
        print(f"{'='*60}")
        print(f"  Successful: {successful}/{len(results)}")
        print(f"  Failed: {len(results)-successful}/{len(results)}")
        
        return results
    
    def save_training_data(self, output_file):
        """
        Save features and energies to CSV for next training iteration.
        
        Args:
            output_file (str): Output CSV filename
        """
        successful_results = [r for r in self.results if r['success']]
        
        if not successful_results:
            print("✗ No successful calculations to save")
            return
        
        features = np.array([r['features'] for r in successful_results])
        energies = np.array([r['e_adsorption'] for r in successful_results])
        
        output = np.c_[features, energies]
        
        np.savetxt(
            output_file,
            output,
            fmt=['%d'] * len(self.metal_types) * len(self.zones) + ['%.5f'],
            delimiter=','
        )
        
        print(f"\n✓ Training data saved to: {output_file}")
        print(f"  Shape: {output.shape}")
        print(f"  Format: {len(self.zones)} zones × {len(self.metal_types)} metals + 1 energy")
        
        return output_file
    
    def resume_from_trajectories(self, traj_dir=None, adsorbate='O'):
        """
        Resume feature extraction from existing trajectory files.
        
        Useful if DFT calculations were already run but features need to be re-extracted.
        
        Args:
            traj_dir (str): Directory containing trajectory files
            adsorbate (str): Adsorbate species
        """
        if traj_dir is None:
            traj_dir = f'traj_b{self.batch_number}'
        
        print(f"\nResuming from trajectories in: {traj_dir}")
        
        results = []
        for config in self.configurations:
            idx = config['index']
            traj_file = f'{traj_dir}/ncf_{idx}{adsorbate}.traj'
            
            try:
                slab_ads = read(traj_file)
                
                features = self.extract_features(slab_ads)
                
                calc = Vasp2(
                    directory=f'vaspcalc_b{self.batch_number}/ncf_{idx}{adsorbate}',
                    restart=True
                )
                e_ads = calc.get_potential_energy()
                
                result = {
                    'index': idx,
                    'features': features,
                    'e_ads': e_ads,
                    'success': True
                }
                
                results.append(result)
                print(f"  ✓ Config {idx}: {e_ads:.4f} eV")
                
            except Exception as e:
                print(f"  ✗ Config {idx}: {e}")
                results.append({'index': idx, 'success': False})
        
        self.results = results
        print(f"\n✓ Resumed {len([r for r in results if r['success']])}/{len(results)} calculations")


def main():
    """Command-line interface for DFT batch runner."""
    parser = argparse.ArgumentParser(description='Run DFT calculations for active learning batch')
    parser.add_argument('--batch', type=int, required=True, help='Batch number')
    parser.add_argument('--metal-file', type=str, help='Metal configuration file (default: batch{N}_metal.csv)')
    parser.add_argument('--start', type=int, default=0, help='Starting index (default: 0)')
    parser.add_argument('--end', type=int, help='Ending index (default: all)')
    parser.add_argument('--adsorbate', type=str, default='O', help='Adsorbate species (default: O)')
    parser.add_argument('--output', type=str, help='Output file (default: DFT_{ads}_batch{N}.csv)')
    parser.add_argument('--resume', action='store_true', help='Resume from existing trajectories')
    
    args = parser.parse_args()
    
    if args.metal_file is None:
        args.metal_file = f'batch{args.batch}_metal.csv'
    
    if args.output is None:
        args.output = f'DFT_{args.adsorbate}_batch{args.batch}.csv'
    
    runner = DFTBatchRunner(batch_number=args.batch)
    
    runner.read_suggestions(args.metal_file, start_idx=args.start, end_idx=args.end)
    
    if args.resume:
        runner.resume_from_trajectories(adsorbate=args.adsorbate)
    else:
        runner.run_all_calculations(adsorbate=args.adsorbate)
    
    runner.save_training_data(args.output)
    
    print(f"\n{'='*60}")
    print("All done!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
