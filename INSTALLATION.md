# Installation and Setup Guide

## Quick Setup (5 minutes)

### Step 1: Copy Your Original Files

You need to copy 2 files from your original code:

```bash
# From your original repository, copy these files:
cp path/to/motif_to_feature.py active_learning/core/features.py
cp path/to/GPRdataspace.py active_learning/core/dataspace.py
```

### Step 2: Update Imports in features.py

Open `active_learning/core/features.py` and change:

```python
# FROM:
from helperMethods import count_atoms

# TO:
from active_learning.utils.helpers import count_atoms
```

### Step 3: Update Imports in dataspace.py

Open `active_learning/core/dataspace.py` and change:

```python
# FROM:
from helperMethods import unique, count_metals, multiplicity
import itertools as it

# TO:
from active_learning.utils.helpers import unique_configurations as unique
from active_learning.utils.helpers import count_metals, multiplicity
import itertools as it
```

Then wrap the main code in a class:

```python
# Add at the end of dataspace.py:

class DataspaceGenerator:
    def __init__(self, n_metals=3, zone_sizes=(3, 6, 3, 3, 3)):
        self.n_metals = n_metals
        self.zone_sizes = zone_sizes
        self.n_configurations = None
    
    def generate(self, filename='GPRdataspace.csv'):
        all_fingerprints(filename, self.n_metals, self.zone_sizes)
        import numpy as np
        data = np.loadtxt(filename, delimiter=',')
        self.n_configurations = len(data)
        print(f"✓ Generated {self.n_configurations} configurations")
```

### Step 4: Install the Package

```bash
cd active_learning_oer
pip install -e .
```

### Step 5: Test

```python
from active_learning import ActiveLearner
print("✓ Package installed successfully!")
```

## What's Already Done

These files are already converted and ready to use:
- ✅ `active_learning/utils/helpers.py` (helperMethods.py)
- ✅ `active_learning/core/gpr.py` (mygaussian.py)
- ✅ `active_learning/core/dft_compatible.py` (possibleFp.py)
- ✅ `active_learning/analysis/activity.py` (activity_plot.py)
- ✅ `active_learning/analysis/processing.py` (sum_element.py)

## What You Need to Do

Just copy and update 2 files:
- `motif_to_feature.py` → `active_learning/core/features.py`
- `GPRdataspace.py` → `active_learning/core/dataspace.py`


## Usage After Installation

```python
from active_learning import Slab, ActiveLearner, ActivityCalculator

# Use exactly like DSTAR
slab = Slab(atoms)
features = slab.features(['Ni', 'Fe', 'Co'])

learner = ActiveLearner()
learner.load_data('DFT_O.csv', 'DFT_OH.csv')
learner.train()
```

See `examples/` for Jupyter notebook tutorials.
