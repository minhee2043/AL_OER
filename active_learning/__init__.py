"""
Active Learning for OER Catalyst Discovery

Example usage:
    >>> from active_learning import Slab, ActiveLearner, ActivityCalculator
    >>> slab = Slab(atoms)
    >>> features = slab.features(['Ni', 'Fe', 'Co'])
"""

__version__ = '1.0.0'

try:
    from active_learning.core.features import Slab
    from active_learning.core.dataspace import DataspaceGenerator
    from active_learning.core.gpr import ActiveLearner, RBF_int
    from active_learning.core.dft_compatible import DFTCompatibleGenerator
    from active_learning.analysis.activity import ActivityCalculator
    from active_learning.analysis.processing import ElementCounter
    
    __all__ = [
        'Slab',
        'DataspaceGenerator',
        'ActiveLearner',
        'RBF_int',
        'DFTCompatibleGenerator',
        'ActivityCalculator',
        'ElementCounter',
    ]
except ImportError as e:
    print(f"Note: Some modules not yet implemented: {e}")
    print("Please copy your code into the placeholder files.")
